import torch
import torch.nn as nn
import torch.nn.functional as F
import math # Added for Kalman initialization

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
    Assumes state is [cx, cy, w, h] and measurement is [cx, cy, w, h].
    """
    def __init__(self, state_dim=4, measurement_dim=4):
        super().__init__()
        self.state_dim = state_dim
        self.measurement_dim = measurement_dim

        # Learnable parameters
        # Initialize transition matrix close to identity
        self.transition_matrix = nn.Parameter(torch.eye(state_dim) + torch.randn(state_dim, state_dim) * 0.01)
        # Initialize observation matrix close to identity
        self.observation_matrix = nn.Parameter(torch.eye(measurement_dim, state_dim) + torch.randn(measurement_dim, state_dim) * 0.01)

        # Learnable noise covariances (learning log variance for stability)
        # Initialize with small values
        self.log_process_noise_diag = nn.Parameter(torch.randn(state_dim) - 2) # Log variance
        self.log_measurement_noise_diag = nn.Parameter(torch.randn(measurement_dim) - 2) # Log variance

        # Identity matrix for update step
        self.register_buffer('identity', torch.eye(self.state_dim))

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
        y = measurement - H @ state_pred # (B, measurement_dim, 1)

        # Calculate innovation covariance: S = H * P_pred * H^T + R
        S = H @ cov_pred @ H.transpose(-1, -2) + R # (B, measurement_dim, measurement_dim)

        # Calculate Kalman gain: K = P_pred * H^T * S^-1
        # Use torch.linalg.solve for potentially better stability than torch.inverse
        try:
            # K = cov_pred @ H.transpose(-1, -2) @ torch.inverse(S)
            S_inv_H_T = torch.linalg.solve(S, H) # Solve S * X = H -> X = S^-1 * H
            K = cov_pred @ S_inv_H_T.transpose(-1, -2) # K = P * H^T * S^-1
        except torch.linalg.LinAlgError:
             # Fallback or handling for singular S
             print("Warning: Singular matrix S encountered in Kalman update. Using pseudo-inverse.")
             K = cov_pred @ H.transpose(-1, -2) @ torch.linalg.pinv(S)


        # Update state: x_updated = x_pred + K * y
        state_updated = state_pred + K @ y # (B, state_dim, 1)

        # Update covariance: P_updated = (I - K * H) * P_pred
        # Joseph form for better numerical stability:
        # P_updated = (I - K H) P_pred (I - K H)^T + K R K^T
        I_KH = self.identity - K @ H # (B, state_dim, state_dim)
        cov_updated = I_KH @ cov_pred @ I_KH.transpose(-1, -2) + K @ R @ K.transpose(-1, -2)
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

        # --- Mamba Sequence Model ---
        self.mamba_dim = config["mamba"]["d_model"]
        # Project encoder output spatial features to mamba dimension
        # We'll use AdaptiveAvgPool3d to handle varying spatial sizes after encoder
        self.encoder_pool = nn.AdaptiveAvgPool3d((None, 1, 1))  # Pool H, W to 1x1
        self.encoder_proj = nn.Linear(self.encoder_output_dim, self.mamba_dim)

        # Project input bounding boxes (4 coords) to mamba dimension
        self.bbox_embedding_dim = config.get(
            "bbox_embedding_dim", 64
        )  # Add flexibility
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

        # --- Mamba Output Head (Now acts as Measurement Generator) ---
        # Predicts 4 coordinates (e.g., cx, cy, w, h) for each frame
        # This is the 'measurement' (z_k) for the Kalman Filter
        self.mamba_measurement_head = nn.Sequential(
            nn.Linear(self.mamba_dim, self.mamba_dim // 2),
            nn.ReLU(),
            nn.Linear(self.mamba_dim // 2, 4),
            nn.Sigmoid(),  # Output normalized coordinates [0, 1]
        )

        # --- Learnable Kalman Filter ---
        self.kalman_filter = LearnableKalmanFilter(state_dim=4, measurement_dim=4)
        # Initial covariance scale (can be tuned or made learnable)
        self.initial_covariance_scale = config.get("kalman_initial_cov_scale", 1e-2)


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
            target_bboxes (torch.Tensor): Ground truth bboxes for ALL T frames (B, T, 4).
                                          Used for teacher-forcing input and KF initialization.
        Returns:
            torch.Tensor: Final predicted bboxes from Kalman Filter (B, T, 4).
            torch.Tensor: Intermediate measurements from Mamba head (B, T, 4).
        """
        # frames: (B, T, C, H, W) - Input video frames
        # target_bboxes: (B, T, 4) - Ground truth bboxes for frames 0 to T-1
        B, T, C_img, H, W = frames.shape
        device = frames.device

        if target_bboxes.shape[1] != T:
             raise ValueError(
                 f"Expected target_bboxes to have T={T} frames, but got {target_bboxes.shape[1]}"
             )
        if target_bboxes.shape[2] != 4:
             raise ValueError(
                 f"Expected target_bboxes to have 4 coordinates, but got {target_bboxes.shape[2]}"
             )

        # --- Prepare Input BBoxes for Mamba (using teacher forcing with target bboxes) ---
        # Use target bboxes from t=0 to T-2 as input for predicting t=1 to T-1
        # For the first time step (t=0), we need an initial bbox input.
        # Let's use the first target bbox itself as input for the first step.
        # Shape: (B, T, 4)
        input_bboxes_for_mamba = target_bboxes # Or potentially shift target_bboxes and pad first element

        # Project input bboxes to Mamba dimension
        # (B, T, 4) -> (B, T, mamba_dim)
        bbox_features = self.bbox_proj(input_bboxes_for_mamba)

        # --- Encode Frames ---
        # Permute for Conv3D: (B, T, C, H, W) -> (B, C, T, H, W)
        x = frames.permute(0, 2, 1, 3, 4)
        for layer in self.encoder_layers:
            x = layer(x)
        # x shape: (B, C_enc, T_enc, H_enc, W_enc)
        # Note: T_enc should be T if strides along T are 1

        # --- Pool and Project Frame Features ---
        # Pool spatial dimensions (H, W)
        # (B, C_enc, T, H_enc, W_enc) -> (B, C_enc, T, 1, 1)
        x_pooled = self.encoder_pool(x)
        # Squeeze spatial dims: -> (B, C_enc, T)
        x_squeezed = x_pooled.squeeze(-1).squeeze(-1)
        # Permute for projection: (B, C_enc, T) -> (B, T, C_enc)
        x_permuted = x_squeezed.permute(0, 2, 1)
        # Project to Mamba dimension: (B, T, C_enc) -> (B, T, mamba_dim)
        frame_features = self.encoder_proj(x_permuted)

        # --- Combine Features and Apply Mamba ---
        # Add frame features and bbox features
        combined_features = frame_features + bbox_features  # Shape: (B, T, mamba_dim)

        # Apply Mamba
        # Input: (B, T, mamba_dim), Output: (B, T, mamba_dim)
        mamba_output = self.mamba(combined_features)

        # --- Get Measurements from Mamba Head ---
        # (B, T, mamba_dim) -> (B, T, 4)
        # These are the 'measurements' (z_k) for the Kalman filter
        mamba_measurements = self.mamba_measurement_head(mamba_output)

        # --- Apply Learnable Kalman Filter ---
        # Initialize Kalman state and covariance
        # Use the first ground truth bbox as the initial state estimate
        kf_state = target_bboxes[:, 0, :].unsqueeze(-1) # Shape: (B, 4, 1)
        # Initialize covariance matrix (diagonal)
        kf_covariance = torch.eye(self.kalman_filter.state_dim, device=device).unsqueeze(0).repeat(B, 1, 1) * self.initial_covariance_scale # Shape: (B, 4, 4)

        kf_predictions_list = []
        for t in range(T):
            # Predict step
            kf_state_pred, kf_cov_pred = self.kalman_filter.predict(kf_state, kf_covariance)

            # Get current measurement from Mamba output
            measurement = mamba_measurements[:, t, :].unsqueeze(-1) # Shape: (B, 4, 1)

            # Update step
            kf_state, kf_covariance = self.kalman_filter.update(kf_state_pred, kf_cov_pred, measurement)

            # Store the updated state (which is the prediction for this time step)
            kf_predictions_list.append(kf_state.squeeze(-1)) # Squeeze back to (B, 4)

        # Concatenate predictions over time
        final_kf_predictions = torch.stack(kf_predictions_list, dim=1) # Shape: (B, T, 4)

        # Return both final KF predictions and intermediate Mamba measurements
        # The loss function can use both.
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
    "kalman_initial_cov_scale": 1e-2, # Initial covariance for KF state
    # Decoder config removed
}


if __name__ == "__main__":
    # Example Usage
    if Mamba is None:
        print("Skipping example: mamba-ssm not installed.")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = TemporalFusionModule(config_tiny_mamba_bbox).to(device)

        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(
            f"Total parameters: {total_params / 1e3:.1f}K"
        )

        B, T, H, W = 2, 5, 64, 64
        frames = torch.randn(B, T, 3, H, W).to(device)  # Input frames (RGB)
        # Ground truth bboxes for ALL T frames, normalized [0,1]
        target_bboxes = torch.rand(B, T, 4).to(device)

        try:
            from loss import KalmanLoss
            loss_criterion = KalmanLoss(kf_loss_weight=1.0, mamba_loss_weight=0.3, loss_type='smooth_l1')
            model.train() # Set model to training mode
            # Get predictions for all T frames
            # Model now returns KF predictions and Mamba measurements
            kf_preds, mamba_meas = model(frames, target_bboxes)

            print(f"KF Predictions shape: {kf_preds.shape}")  # Should be (B, T, 4)
            print(f"Mamba Measurements shape: {mamba_meas.shape}") # Should be (B, T, 4)

            # Calculate loss
            total_loss, loss_components = loss_criterion(kf_preds, mamba_meas, target_bboxes)
            print(f"Total Loss: {total_loss.item():.4f}")
            print(f"Loss Components: {loss_components}")



        except ImportError as e:
            print(f"Import Error during execution: {e}")
        except Exception as e:
            print(f"An error occurred during model execution: {e}")
            import traceback
            traceback.print_exc() # Print detailed traceback
