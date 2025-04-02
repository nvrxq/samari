import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalFusionModule(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.skip_connections = config["encoder"]["skip_connections"]

        self.encoder_layers = self._build_encoder()
        self.encoder_channels = config["encoder"]["channels"]

        # Временной трансформер
        self.temporal_transformer = TemporalTransformer(
            d_model=config["transformer"]["d_model"],
            nhead=config["transformer"]["nhead"],
            num_layers=config["transformer"]["num_layers"],
            dropout=config["transformer"]["dropout"],
        )

        self.decoder_layers = self._build_decoder()
        if self.skip_connections:
            self.adapters = self._build_adapters()

        self.mask_convs = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv3d(ch, 1, kernel_size=(1, 3, 3), padding=(0, 1, 1)),
                    nn.Sigmoid(),
                )
                for ch in config["decoder"]["channels"]  # Используем каналы декодера
            ]
        )

    def _build_encoder(self):
        layers = []
        cfg = self.config["encoder"]
        in_channels = cfg["input_channels"]

        for i, out_channels in enumerate(cfg["channels"]):
            block = [
                nn.Conv3d(
                    in_channels,
                    out_channels,
                    kernel_size=cfg["kernels"][i],
                    stride=cfg["strides"][i],
                    padding=cfg["paddings"][i],
                ),
                nn.BatchNorm3d(out_channels),
                nn.ReLU(),
                nn.MaxPool3d(
                    cfg["pool_kernels"][i],
                    padding=cfg.get("pool_paddings", [0] * len(cfg["pool_kernels"]))[i],
                ),
            ]
            layers.append(nn.Sequential(*block))
            in_channels = out_channels

        return nn.ModuleList(layers)

    def _build_decoder(self):
        layers = []
        cfg = self.config["decoder"]
        in_channels = self.config["transformer"]["d_model"]

        for i, out_ch in enumerate(cfg["channels"]):
            block = [
                nn.ConvTranspose3d(
                    in_channels,
                    out_ch,
                    kernel_size=cfg["kernels"][i],
                    stride=cfg["strides"][i],
                    padding=cfg["paddings"][i],
                    output_padding=cfg["output_paddings"][i],
                ),
                nn.BatchNorm3d(out_ch),
                nn.ReLU(),
                SelfAttention3D(out_ch),
            ]
            layers.append(nn.Sequential(*block))
            in_channels = out_ch

        return nn.ModuleList(layers)

    def _build_adapters(self):
        """Строит адаптеры для skip-соединений."""
        adapters = []
        decoder_cfg = self.config["decoder"]
        skip_channels = list(reversed(self.encoder_channels))
        decoder_channels = decoder_cfg["channels"]

        num_skips = min(len(skip_channels), len(decoder_channels))

        for i in range(num_skips):
            adapter = nn.Sequential(
                nn.Conv3d(skip_channels[i], decoder_channels[i], kernel_size=1),
                nn.Upsample(
                    scale_factor=(1, 2, 2), mode="trilinear", align_corners=True
                ),
            )
            adapters.append(adapter)
        return nn.ModuleList(adapters)

    def forward(self, frames, past_masks):
        # frames: (B, T, C, H, W)
        # past_masks: (B, T-1, 1, H, W) - Ground truth for frames 0 to T-2
        B, T, C, H, W = frames.shape
        device = frames.device

        # Create a placeholder mask for the T-th frame (the one we want to predict)
        # This placeholder is concatenated with the *input* frames, not the ground truth masks.
        # Shape: (B, 1, 1, H, W)
        future_mask_placeholder = torch.zeros((B, 1, 1, H, W), device=device)

        # Check if past_masks has the expected T-1 dimension
        if past_masks.shape[1] != T - 1:
             raise ValueError(f"Expected past_masks to have T-1={T-1} frames, but got {past_masks.shape[1]}")

        # Concatenate the known past masks with the placeholder for the future mask
        # This creates the input mask sequence for the model.
        # Shape: (B, T-1, 1, H, W) + (B, 1, 1, H, W) -> (B, T, 1, H, W)
        input_masks = torch.cat([past_masks, future_mask_placeholder], dim=1) # Concatenating along the T dimension

        # Combine frames and input masks along the channel dimension
        # frames: (B, T, C, H, W)
        # input_masks: (B, T, 1, H, W)
        # Result shape: (B, T, C+1, H, W)
        # Permute for Conv3D: (B, C+1, T, H, W)
        combined = torch.cat([frames, input_masks], dim=2).permute(0, 2, 1, 3, 4)

        encoder_features = []
        x = combined
        for layer in self.encoder_layers:
            x = layer(x)
            encoder_features.append(x)

        # --- Store spatial dimensions before transformer ---
        B_enc, C_enc, T_enc, H_enc, W_enc = x.shape # Get shape after last encoder layer

        # --- Reshape for Transformer ---
        # (B, C, T, H, W) -> (B, T, H, W, C) -> (B*H*W, T, C)
        x = x.permute(0, 2, 3, 4, 1).reshape(B_enc * H_enc * W_enc, T_enc, C_enc)
        x = self.temporal_transformer(x) # Output shape (B*H*W, T, C)

        # --- Reshape back and potentially interpolate Time ---
        # (B*H*W, T, C) -> (B, H, W, T, C) -> (B, C, T, H, W)
        x = x.view(B_enc, H_enc, W_enc, T_enc, C_enc).permute(0, 4, 3, 1, 2)

        # Interpolate Time dimension if T_enc != T (e.g., if encoder subsampled T)
        # The current config doesn't seem to subsample T in the encoder, so T_enc should be T.
        # If it did, interpolation would be needed here.
        if x.shape[2] != T:
             x = F.interpolate(
                 x, size=(T, H_enc, W_enc), mode="trilinear", align_corners=False # Use align_corners=False generally
             )

        masks = []
        num_decoder_layers = len(self.decoder_layers)
        num_mask_convs = len(self.mask_convs)

        # --- Decoder ---
        encoder_features.reverse() # Reverse for standard U-Net style skip connections
        adapters = self.adapters # No need to reverse adapters if built corresponding to reversed features

        for i, layer in enumerate(self.decoder_layers):
            # Apply skip connection *before* the decoder layer (common U-Net pattern)
            if self.skip_connections and i < len(adapters):
                skip = encoder_features[i] # Get corresponding encoder feature
                adapted_skip = adapters[i](skip) # Adapt channels and potentially upsample H, W

                # Ensure skip connection spatial dimensions match x before concatenation/addition
                if x.shape[-3:] != adapted_skip.shape[-3:]:
                    # Interpolate skip connection T, H, W to match x's T, H, W
                    adapted_skip = F.interpolate(
                        adapted_skip,
                        size=x.shape[-3:], # Match T, H, W of the current decoder feature map x
                        mode="trilinear",
                        align_corners=False, # Use align_corners=False generally
                    )
                # Add or concatenate skip connection
                # Using addition here, ensure channels match (adapter should handle this)
                x = x + adapted_skip

            # Apply decoder layer (Upsampling + Conv + BN + ReLU + Attention)
            x = layer(x)

            # Generate mask prediction at this decoder level
            if i < num_mask_convs:
                # mask_convs expects input channels matching decoder output channels
                mask = self.mask_convs[i](x) # Output: (B, 1, T, H_dec, W_dec)
                masks.append(mask)

        # --- Final Output Mask ---
        # Usually, the mask from the last decoder layer (highest resolution) is used.
        # Upsample the final mask to the original input H, W. T should already match.
        final_mask_raw = masks[-1] # Shape (B, 1, T, H_final_dec, W_final_dec)
        output_mask = F.interpolate(
             final_mask_raw, size=(T, H, W), mode="trilinear", align_corners=False # Match original T, H, W
        ) # Shape: (B, 1, T, H, W)

        # Permute to match expected output: (B, T, 1, H, W)
        # The loss function expects predictions for all T frames.
        return output_mask.permute(0, 2, 1, 3, 4) # Shape: (B, T, 1, H, W)


class TemporalTransformer(nn.Module):
    def __init__(self, d_model, nhead, num_layers, dropout=0.1):
        super().__init__()
        self.positional_encoding = PositionalEncoding3D(d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x):
        x = self.positional_encoding(x)
        return self.transformer(x)


class PositionalEncoding3D(nn.Module):
    def __init__(self, d_model, max_len=500):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float()
            * (-torch.log(torch.tensor(10000.0)) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, : x.size(1)]


class SelfAttention3D(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.query = nn.Conv3d(channels, channels // 8, kernel_size=1)
        self.key = nn.Conv3d(channels, channels // 8, kernel_size=1)
        self.value = nn.Conv3d(channels, channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        B, C, T, H, W = x.shape
        proj_query = (
            self.query(x).view(B, -1, T * H * W).permute(0, 2, 1)
        )  # [B, THW, C']
        proj_key = self.key(x).view(B, -1, T * H * W)  # [B, C', THW]
        energy = torch.bmm(proj_query, proj_key)  # [B, THW, THW]
        attention = self.softmax(energy)
        proj_value = self.value(x).view(B, -1, T * H * W)  # [B, C, THW]
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(B, C, T, H, W)
        return self.gamma * out + x


config_tiny = {
    "encoder": {
        "input_channels": 4,  # 3 (RGB) + 1 (Mask)
        "skip_connections": True,
        "channels": [16, 32, 64],
        "kernels": [(3, 3, 3)] * 3,
        "strides": [(1, 1, 1)] * 3,  # Не меняем T
        "paddings": [(1, 1, 1)] * 3,
        "pool_kernels": [(1, 2, 2)] * 3,  # Уменьшаем H, W
        "pool_paddings": [(0, 0, 0)] * 3,
    },
    "transformer": {
        "d_model": 64,  # Должно совпадать с последним каналом энкодера
        "nhead": 4,
        "num_layers": 2,
        "dropout": 0.1,
    },
    "decoder": {
        "channels": [64, 32, 16],  # Выходные каналы декодера
        # "mask_channels" удалено, т.к. используем "channels"
        "kernels": [(3, 3, 3)] * 3,
        "strides": [(1, 2, 2)] * 3,  # Увеличиваем H, W
        "paddings": [(1, 1, 1)] * 3,
        "output_paddings": [(0, 1, 1)] * 3,  # Для компенсации stride=2
    },
}

if __name__ == "__main__":
    from torchviz import make_dot

    model = TemporalFusionModule(config_tiny)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params / 1e3:.1f}K")  # ~364K

    frames = torch.randn(2, 5, 3, 64, 64)
    masks = torch.rand(2, 4, 1, 64, 64)
    try:
        output = model(frames, masks)
        dot = make_dot(output, params=dict(model.named_parameters()))

        file_format = "png"
        output_filename = f"tfn_architecture.{file_format}"
        dot.render(output_filename.replace(f".{file_format}", ""), format=file_format)

        print(f"Архитектура модели сохранена в файл: {output_filename}")
        print("Для просмотра откройте этот файл.")

    except ImportError as e:
        print(f"Ошибка импорта: {e}")
        print("Убедитесь, что библиотеки torchviz и graphviz установлены.")
        print(
            "Также убедитесь, что сама программа Graphviz установлена в вашей системе и доступна в PATH."
        )
    except RuntimeError as e:
        print(f"Ошибка во время выполнения: {e}")
        print("Возможно, проблема с установкой Graphviz или его доступностью в PATH.")
    except Exception as e:
        print(f"Неожиданная ошибка: {e}")
