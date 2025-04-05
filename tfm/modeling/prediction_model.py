import torch
import torch.nn as nn
import torch.nn.functional as F
import math  # Added for Kalman initialization

# Try importing Mamba; handle potential ImportError
try:
    from mamba_ssm import Mamba
except ImportError:
    Mamba = None
    print("Warning: mamba-ssm package not found. Please install it to use Mamba.")


# +++ Learnable Kalman Filter Module +++
class LearnableKalmanFilter(nn.Module):
    """
    Learnable Kalman Filter for bounding box tracking.
    State includes position, size, and velocities: [cx, cy, w, h, vx, vy, vw, vh].
    Measurement is position and size: [cx, cy, w, h].
    """

    def __init__(self, state_dim=8, measurement_dim=4):
        super().__init__()
        self.state_dim = state_dim
        self.measurement_dim = measurement_dim

        # Learnable parameters
        # Initialize transition matrix (F) for constant velocity (dt=1)
        # x_new = x + vx*1, vx_new = vx, etc.
        F = torch.eye(state_dim)
        for i in range(state_dim // 2):  # Set velocity updates for cx, cy, w, h
            F[i, i + state_dim // 2] = 1.0
        self.transition_matrix = nn.Parameter(
            F + torch.randn(state_dim, state_dim) * 0.001
        )  # Smaller noise

        # Initialize observation matrix (H) to select [cx, cy, w, h] from state
        H = torch.zeros(measurement_dim, state_dim)
        for i in range(measurement_dim):
            H[i, i] = 1.0
        self.observation_matrix = nn.Parameter(
            H + torch.randn(measurement_dim, state_dim) * 0.01
        )

        # Learnable noise covariances (learning log variance for stability)
        # Initialize with small values, potentially different scales for pos/vel
        log_process_noise_pos = (
            torch.randn(state_dim // 2) - 2
        )  # Log variance for pos/size
        log_process_noise_vel = (
            torch.randn(state_dim // 2) - 4
        )  # Log variance for velocity (smaller initial noise)
        self.log_process_noise_diag = nn.Parameter(
            torch.cat([log_process_noise_pos, log_process_noise_vel])
        )
        self.log_measurement_noise_diag = nn.Parameter(
            torch.randn(measurement_dim) - 2
        )  # Log variance

        # Identity matrix for update step
        self.register_buffer("identity", torch.eye(self.state_dim))

    def _get_cov_matrix(self, log_diag):
        # Ensure positivity by exponentiating log variance
        # Add small epsilon for numerical stability
        return torch.diag(torch.exp(log_diag) + 1e-6)

    def predict(self, state, covariance):
        """Kalman Filter prediction step."""
        F = self.transition_matrix
        Q = self._get_cov_matrix(self.log_process_noise_diag)

        # Predict state: x_pred = F * x
        # state shape: (B, state_dim, 1)
        state_pred = F @ state
        # Predict covariance: P_pred = F * P * F^T + Q
        # covariance shape: (B, state_dim, state_dim)
        cov_pred = F @ covariance @ F.transpose(-1, -2) + Q

        return state_pred, cov_pred

    def update(self, state_pred, cov_pred, measurement):
        """Kalman Filter update step."""
        H = self.observation_matrix
        R = self._get_cov_matrix(self.log_measurement_noise_diag)

        # measurement shape: (B, measurement_dim, 1)
        # state_pred shape: (B, state_dim, 1)
        # cov_pred shape: (B, state_dim, state_dim)

        # Calculate innovation: y = z - H * x_pred
        y = measurement - H @ state_pred  # (B, measurement_dim, 1)

        # Calculate innovation covariance: S = H * P_pred * H^T + R
        S = (
            H @ cov_pred @ H.transpose(-1, -2) + R
        )  # (B, measurement_dim, measurement_dim)

        # Calculate Kalman gain: K = P_pred * H^T * S^-1
        # Use torch.linalg.solve for potentially better stability than torch.inverse
        try:
            # K = cov_pred @ H.transpose(-1, -2) @ torch.inverse(S)
            S_inv_H_T = torch.linalg.solve(S, H)  # Solve S * X = H -> X = S^-1 * H
            K = cov_pred @ S_inv_H_T.transpose(-1, -2)  # K = P * H^T * S^-1
        except torch.linalg.LinAlgError:
            # Fallback or handling for singular S
            print(
                "Warning: Singular matrix S encountered in Kalman update. Using pseudo-inverse."
            )
            K = cov_pred @ H.transpose(-1, -2) @ torch.linalg.pinv(S)

        # Update state: x_updated = x_pred + K * y
        state_updated = state_pred + K @ y  # (B, state_dim, 1)

        # Update covariance: P_updated = (I - K * H) * P_pred
        # Joseph form for better numerical stability:
        # P_updated = (I - K H) P_pred (I - K H)^T + K R K^T
        I_KH = self.identity - K @ H  # (B, state_dim, state_dim)
        cov_updated = I_KH @ cov_pred @ I_KH.transpose(-1, -2) + K @ R @ K.transpose(
            -1, -2
        )
        # cov_updated = (self.identity - K @ H) @ cov_pred # Simpler form

        return state_updated, cov_updated


class TemporalFusionModule(nn.Module):
    def __init__(self, config):
        super().__init__()
        if Mamba is None:
            raise ImportError(
                "Mamba module requires mamba-ssm package. Please install it."
            )

        self.config = config
        self.encoder_layers = self._build_encoder()
        self.encoder_output_dim = config["encoder"]["channels"][-1]

        # --- Determine Encoder Output Shape (for Flattening - Пункт 4) ---
        # We need H_enc, W_enc after the encoder. Pass a dummy input.
        # Use a typical input size expected by the model. Adjust if needed.
        dummy_input_shape = config.get(
            "dummy_input_shape", (1, 3, config.get("max_frames", 5), 256, 256)
        )  # B, C, T, H, W
        with torch.no_grad():
            dummy_input = torch.zeros(dummy_input_shape)
            dummy_output = dummy_input
            for layer in self.encoder_layers:
                dummy_output = layer(dummy_output)
            # dummy_output shape: (B, C_enc, T_enc, H_enc, W_enc)
            self.encoder_output_spatial_dims = dummy_output.shape[-2:]  # (H_enc, W_enc)
            self.flattened_encoder_dim = (
                self.encoder_output_dim
                * self.encoder_output_spatial_dims[0]
                * self.encoder_output_spatial_dims[1]
            )
            print(f"Encoder output spatial dims: {self.encoder_output_spatial_dims}")
            print(f"Flattened encoder feature dimension: {self.flattened_encoder_dim}")

        # --- Mamba Sequence Model ---
        self.mamba_dim = config["mamba"]["d_model"]

        # --- Feature Projections ---
        # Project FLATTENED encoder output features to mamba dimension (Пункт 4)
        # self.encoder_pool = nn.AdaptiveAvgPool3d((None, 1, 1)) # Removed
        self.encoder_proj = nn.Linear(
            self.flattened_encoder_dim, self.mamba_dim
        )  # Updated input dim

        # Project input bounding boxes (4 coords) to mamba dimension
        self.bbox_embedding_dim = config.get("bbox_embedding_dim", 64)
        self.bbox_proj = nn.Sequential(
            nn.Linear(4, self.bbox_embedding_dim),
            nn.ReLU(),
            nn.Linear(self.bbox_embedding_dim, self.mamba_dim),
        )

        # Mamba layer
        self.mamba = Mamba(
            d_model=self.mamba_dim,
            d_state=config["mamba"]["d_state"],
            d_conv=config["mamba"]["d_conv"],
            expand=config["mamba"]["expand"],
        )

        # --- Kalman Filter (Пункт 3) ---
        self.kalman_state_dim = config.get("kalman_state_dim", 8)  # Default to 8D state
        self.kalman_measurement_dim = 4  # Measurement is still [cx, cy, w, h]
        self.kalman_filter = LearnableKalmanFilter(
            state_dim=self.kalman_state_dim, measurement_dim=self.kalman_measurement_dim
        )
        self.initial_covariance_scale = config.get("kalman_initial_cov_scale", 1e-2)

        # --- Output Head ---
        # Predicts the 'measurement' (z_k) for Kalman from Mamba output
        self.mamba_measurement_head = nn.Linear(
            self.mamba_dim, self.kalman_measurement_dim
        )  # Output dim = 4

    def _build_encoder(self):
        layers = []
        cfg = self.config["encoder"]
        # Input channels are just image channels now (e.g., 3 for RGB)
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

    def forward(self, frames, target_bboxes):
        """
        Args:
            frames (torch.Tensor): Input video frames (B, T, C, H, W).
            target_bboxes (torch.Tensor): Ground truth bounding boxes (B, T, 4),
                                          normalized [cx, cy, w, h]. Used for teacher
                                          forcing and initial KF state.

        Returns:
            torch.Tensor: Final predicted bounding boxes from Kalman Filter (B, T, 4).
                          (Only position/size part of the state).
            torch.Tensor: Intermediate measurements predicted by Mamba head (B, T, 4).
        """
        B, T, C, H, W = frames.shape
        device = frames.device

        # --- Prepare Input BBoxes for Mamba (Shifted Teacher Forcing - from previous step) ---
        if T > 1:
            shifted_bboxes = target_bboxes[:, :-1, :]
            first_bbox_input = target_bboxes[:, 0:1, :]
            input_bboxes_for_mamba = torch.cat(
                [first_bbox_input, shifted_bboxes], dim=1
            )
        else:
            input_bboxes_for_mamba = target_bboxes
        bbox_features = self.bbox_proj(input_bboxes_for_mamba)  # (B, T, mamba_dim)

        # --- Encode Frames ---
        x = frames.permute(0, 2, 1, 3, 4)  # (B, C, T, H, W)
        for layer in self.encoder_layers:
            x = layer(x)
        # x shape: (B, C_enc, T_enc, H_enc, W_enc)

        # --- Flatten and Project Frame Features (Пункт 4) ---
        # x_pooled = self.encoder_pool(x) # Removed
        # x_squeezed = x_pooled.squeeze(-1).squeeze(-1) # Removed
        # Flatten spatial dims H_enc, W_enc
        # (B, C_enc, T, H_enc, W_enc) -> (B, C_enc, T, H_enc * W_enc)
        x_flattened_spatial = x.flatten(start_dim=3)
        # Permute for projection: (B, C_enc, T, H_enc*W_enc) -> (B, T, C_enc * H_enc * W_enc)
        x_permuted = x_flattened_spatial.permute(0, 2, 1, 3).flatten(start_dim=2)
        # Project to Mamba dimension: (B, T, flattened_dim) -> (B, T, mamba_dim)
        frame_features = self.encoder_proj(x_permuted)

        # --- Combine Features and Apply Mamba ---
        combined_features = frame_features + bbox_features
        mamba_output = self.mamba(combined_features)  # (B, T, mamba_dim)

        # --- Get Measurements from Mamba Head ---
        mamba_measurements = self.mamba_measurement_head(mamba_output)  # (B, T, 4)

        # --- Apply Learnable Kalman Filter (Пункт 3) ---
        # Initialize Kalman state and covariance
        initial_state_pos = target_bboxes[:, 0, :]  # (B, 4) - cx, cy, w, h
        # Estimate initial velocity (vx, vy, vw, vh)
        if T > 1:
            # Simple difference between first two frames
            initial_velocity = target_bboxes[:, 1, :] - target_bboxes[:, 0, :]  # (B, 4)
        else:
            # Assume zero initial velocity if only one frame
            initial_velocity = torch.zeros_like(initial_state_pos)

        # Combine position and velocity for initial state
        initial_state = torch.cat(
            [initial_state_pos, initial_velocity], dim=1
        )  # (B, 8)
        kf_state = initial_state.unsqueeze(-1)  # Shape: (B, 8, 1)

        # Initialize covariance matrix (diagonal)
        kf_covariance = (
            torch.eye(self.kalman_state_dim, device=device).unsqueeze(0).repeat(B, 1, 1)
            * self.initial_covariance_scale
        )  # Shape: (B, 8, 8)

        kf_predictions_list = (
            []
        )  # Stores the position/size part of the state [cx, cy, w, h]
        for t in range(T):
            # Predict step
            kf_state_pred, kf_cov_pred = self.kalman_filter.predict(
                kf_state, kf_covariance
            )

            # Get current measurement from Mamba output
            measurement = mamba_measurements[:, t, :].unsqueeze(-1)  # Shape: (B, 4, 1)

            # Update step
            kf_state, kf_covariance = self.kalman_filter.update(
                kf_state_pred, kf_cov_pred, measurement
            )

            # Store the position/size part [cx, cy, w, h] of the updated state
            kf_predictions_list.append(
                kf_state[:, : self.kalman_measurement_dim, 0]
            )  # Shape (B, 4)

        # Concatenate predictions over time
        final_kf_predictions = torch.stack(
            kf_predictions_list, dim=1
        )  # Shape: (B, T, 4)

        # Return KF predictions (pos/size only) and Mamba measurements
        return final_kf_predictions, mamba_measurements


# Remove TemporalTransformer, PositionalEncoding3D, SelfAttention3D classes


# --- Updated Config ---
config_tiny_mamba_bbox = {
    "encoder": {
        "input_channels": 3,  # Just RGB
        "channels": [16, 32, 64],
        "kernels": [(3, 3, 3)] * 3,
        "strides": [(1, 1, 1)] * 3,  # Keep T dimension
        "paddings": [(1, 1, 1)] * 3,
        "pool_kernels": [(1, 2, 2)] * 3,  # Pool H, W
        "pool_paddings": [(0, 0, 0)] * 3,
    },
    "mamba": {
        "d_model": 128,  # Dimension for Mamba layer
        "d_state": 16,
        "d_conv": 4,
        "expand": 2,
    },
    "bbox_embedding_dim": 64,  # Intermediate dim for bbox projection
    "kalman_state_dim": 8,  # Explicitly set state dimension
    "kalman_initial_cov_scale": 1e-2,  # Initial covariance for KF state
    "dummy_input_shape": (1, 3, 10, 256, 256),  # Example: B=1, T=10, H/W=256
    # Decoder config removed
}


if __name__ == "__main__":
    # Example Usage
    if Mamba is None:
        print("Skipping example: mamba-ssm not installed.")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Use the config that defines the expected input shape
        config_to_use = config_tiny_mamba_bbox_v2  # Or your latest config
        model = TemporalFusionModule(config_to_use).to(device)

        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Total parameters: {total_params / 1e3:.1f}K")

        # --- Use input dimensions consistent with the config's dummy_input_shape ---
        dummy_shape = config_to_use.get("dummy_input_shape", (1, 3, 5, 256, 256))
        B = 2  # Example batch size
        T = dummy_shape[2]  # Get T from dummy shape
        H = dummy_shape[3]  # Get H from dummy shape
        W = dummy_shape[4]  # Get W from dummy shape
        print(f"Using test input shape: B={B}, T={T}, H={H}, W={W}")

        frames = torch.randn(B, T, 3, H, W).to(device)  # Input frames (RGB)
        # Ground truth bboxes for ALL T frames, normalized [0,1]
        target_bboxes = torch.rand(B, T, 4).to(device)

        try:
            # --- Import Loss (Make sure loss.py is accessible) ---
            import sys
            import os

            # Add parent directory to sys.path if loss.py is in the parent dir
            current_dir = os.path.dirname(os.path.abspath(__file__))
            parent_dir = os.path.dirname(current_dir)
            if parent_dir not in sys.path:
                sys.path.append(parent_dir)
            from loss import KalmanLoss

            # --- End Import ---

            # Use a loss type supported by your updated KalmanLoss
            loss_criterion = KalmanLoss(
                kf_loss_weight=1.0, mamba_loss_weight=0.3, loss_type="smooth_l1"
            )  # Or 'giou', 'l1', 'mse'
            model.train()  # Set model to training mode
            # Get predictions for all T frames
            # Model now returns KF predictions and Mamba measurements
            kf_preds, mamba_meas = model(frames, target_bboxes)

            print(f"KF Predictions shape: {kf_preds.shape}")  # Should be (B, T, 4)
            print(
                f"Mamba Measurements shape: {mamba_meas.shape}"
            )  # Should be (B, T, 4)

            # Calculate loss
            total_loss, loss_components = loss_criterion(
                kf_preds, mamba_meas, target_bboxes
            )
            print(f"Total Loss: {total_loss.item():.4f}")
            print(f"Loss Components: {loss_components}")

        except ImportError as e:
            print(f"Import Error during execution: {e}")
            print("Ensure loss.py is in the correct path (e.g., parent directory).")
        except Exception as e:
            print(f"An error occurred during model execution: {e}")
            import traceback

            traceback.print_exc()  # Print detailed traceback
