import torch
import torch.nn as nn # Added for KalmanLoss
import torch.nn.functional as F


def weighted_bce_loss(predicted_masks, target_masks, last_frame_weight=2.0):
    """
    Calculates Binary Cross Entropy loss with a higher weight for the last frame.
    (Used for mask prediction)

    Args:
        predicted_masks (torch.Tensor): Predicted masks from the model.
                                         Shape: (B, N, 1, H, W)
        target_masks (torch.Tensor): Ground truth masks.
                                      Shape: (B, N, 1, H, W)
        last_frame_weight (float): The weight multiplier for the loss
                                   of the last frame in the sequence.

    Returns:
        torch.Tensor: The calculated weighted BCE loss (scalar).
    """
    B, N, C, H, W = predicted_masks.shape
    assert C == 1, "Masks should have a single channel"
    assert (
        predicted_masks.shape == target_masks.shape
    ), f"Predicted masks shape {predicted_masks.shape} does not match target masks shape {target_masks.shape}"
    assert N > 0, "Number of frames (N) must be greater than 0"

    # Ensure masks are float and targets are float for BCE
    predicted_masks = predicted_masks.float()
    target_masks = target_masks.float()

    # Calculate BCE loss per pixel, without reduction
    # Reshape to (B*N, 1, H, W) for F.binary_cross_entropy
    bce_loss = F.binary_cross_entropy(
        predicted_masks.view(B * N, C, H, W),
        target_masks.view(B * N, C, H, W),
        reduction="none",
    )

    # Average loss over spatial dimensions (H, W) and channel (C)
    loss_per_frame = bce_loss.mean(dim=[1, 2, 3])  # Shape: (B*N)

    # Reshape back to (B, N) to apply weights per frame
    loss_per_frame = loss_per_frame.view(B, N)

    # Create weights: higher weight for the last frame
    weights = torch.ones(N, device=predicted_masks.device)
    if N > 0:  # Apply weight only if there are frames
        weights[-1] = last_frame_weight

    # Apply weights to the loss of each frame
    # Shape: (B, N) * (N,) -> (B, N) using broadcasting
    weighted_losses = loss_per_frame * weights

    # Calculate the mean loss across frames for each batch item,
    # effectively averaging with the applied weights.
    # The division by sum(weights) normalizes the loss.
    loss_per_batch_item = weighted_losses.sum(dim=1) / weights.sum()  # Shape: (B,)

    # Calculate the final mean loss over the batch
    final_loss = loss_per_batch_item.mean()

    return final_loss


def weighted_bbox_l1_loss(predicted_bboxes, target_bboxes, last_frame_weight=2.0):
    """
    Calculates L1 loss (Mean Absolute Error) between predicted and target
    bounding boxes, with an optional higher weight for the last frame.

    Args:
        predicted_bboxes (torch.Tensor): Predicted bounding boxes from the model.
                                         Shape: (B, T, 4) - Normalized [0, 1]
        target_bboxes (torch.Tensor): Ground truth bounding boxes.
                                      Shape: (B, T, 4) - Normalized [0, 1]
        last_frame_weight (float): The weight multiplier for the loss
                                   of the last frame in the sequence.

    Returns:
        torch.Tensor: The calculated weighted L1 loss (scalar).
    """
    B, T, C = predicted_bboxes.shape
    assert C == 4, "Bounding boxes should have 4 coordinates"
    assert (
        predicted_bboxes.shape == target_bboxes.shape
    ), f"Predicted bboxes shape {predicted_bboxes.shape} does not match target bboxes shape {target_bboxes.shape}"
    assert T > 0, "Number of frames (T) must be greater than 0"

    # Ensure tensors are float
    predicted_bboxes = predicted_bboxes.float()
    target_bboxes = target_bboxes.float()

    # Calculate L1 loss per coordinate, without reduction
    # Shape: (B, T, 4)
    l1_loss = torch.abs(predicted_bboxes - target_bboxes)

    # Sum loss across the 4 coordinates for each box
    # Shape: (B, T)
    loss_per_frame = l1_loss.sum(dim=2)

    # Create weights: higher weight for the last frame
    weights = torch.ones(T, device=predicted_bboxes.device)
    if T > 0:  # Apply weight only if there are frames
        weights[-1] = last_frame_weight

    # Apply weights to the loss of each frame
    # Shape: (B, T) * (T,) -> (B, T) using broadcasting
    weighted_losses = loss_per_frame * weights

    # Calculate the mean loss across frames for each batch item,
    # effectively averaging with the applied weights.
    # The division by sum(weights) normalizes the loss.
    loss_per_batch_item = weighted_losses.sum(dim=1) / weights.sum()  # Shape: (B,)

    # Calculate the final mean loss over the batch
    final_loss = loss_per_batch_item.mean()

    return final_loss


# +++ Kalman Loss Function (Moved from model.py) +++
class KalmanLoss(nn.Module):
    def __init__(self, kf_loss_weight=1.0, mamba_loss_weight=0.2, loss_type='smooth_l1'):
        """
        Loss function for the Kalman Filter enhanced model.

        Args:
            kf_loss_weight (float): Weight for the loss on the final Kalman Filter predictions.
            mamba_loss_weight (float): Weight for the loss on the intermediate Mamba measurements.
                                       Helps guide the Mamba feature extraction.
            loss_type (str): Type of regression loss ('l1', 'smooth_l1', 'mse').
        """
        super().__init__()
        self.kf_loss_weight = kf_loss_weight
        self.mamba_loss_weight = mamba_loss_weight
        if loss_type == 'smooth_l1':
            # Use beta=1.0 for standard Smooth L1
            self.loss_fn = lambda pred, target: F.smooth_l1_loss(pred, target, beta=1.0, reduction='mean')
        elif loss_type == 'l1':
            self.loss_fn = lambda pred, target: F.l1_loss(pred, target, reduction='mean')
        elif loss_type == 'mse':
            self.loss_fn = lambda pred, target: F.mse_loss(pred, target, reduction='mean')
        else:
            raise ValueError(f"Unsupported loss_type: {loss_type}")

    def forward(self, kf_predictions, mamba_measurements, target_bboxes):
        """
        Calculate the combined loss.

        Args:
            kf_predictions (torch.Tensor): Final predictions from Kalman Filter (B, T, 4).
            mamba_measurements (torch.Tensor): Intermediate measurements from Mamba head (B, T, 4).
            target_bboxes (torch.Tensor): Ground truth bounding boxes (B, T, 4).

        Returns:
            torch.Tensor: The total calculated loss.
            dict: Dictionary containing individual loss components.
        """
        # Ensure target_bboxes are float
        target_bboxes = target_bboxes.float()

        # Calculate loss for Kalman Filter predictions vs targets
        loss_kf = self.loss_fn(kf_predictions, target_bboxes)

        # Calculate loss for Mamba measurements vs targets
        loss_mamba = self.loss_fn(mamba_measurements, target_bboxes)

        # Combine losses with weights
        total_loss = (self.kf_loss_weight * loss_kf +
                      self.mamba_loss_weight * loss_mamba)

        loss_dict = {
            'total_loss': total_loss,
            'loss_kf': loss_kf,
            'loss_mamba': loss_mamba
        }
        return total_loss, loss_dict


# Example Usage (can be removed or kept for testing)
if __name__ == "__main__":
    # --- BCE Loss Example ---
    B, N, H, W = 2, 5, 1, 1  # Batch size 2, 5 frames, 1x1 resolution for simplicity
    preds_mask = torch.rand(B, N, 1, H, W)
    targets_mask = torch.rand(B, N, 1, H, W)
    loss_bce = weighted_bce_loss(preds_mask, targets_mask, last_frame_weight=3.0)
    print(f"Calculated Weighted BCE Loss: {loss_bce.item()}")

    # --- BBox L1 Loss Example ---
    B, T = 2, 5  # Batch size 2, 5 frames (T includes the target frame)
    # Example predictions (normalized coordinates)
    preds_bbox = torch.rand(B, T, 4)
    # Example targets (normalized coordinates)
    targets_bbox = torch.rand(B, T, 4)

    loss_l1 = weighted_bbox_l1_loss(preds_bbox, targets_bbox, last_frame_weight=3.0)
    print(f"Calculated Weighted BBox L1 Loss: {loss_l1.item()}")

    # Test with T=1
    preds_bbox_single = torch.rand(B, 1, 4)
    targets_bbox_single = torch.rand(B, 1, 4)
    loss_l1_single = weighted_bbox_l1_loss(
        preds_bbox_single, targets_bbox_single, last_frame_weight=3.0
    )
    print(f"Calculated Weighted BBox L1 Loss (T=1): {loss_l1_single.item()}")

    # Compare with standard L1
    standard_l1 = F.l1_loss(preds_bbox, targets_bbox)
    # Note: Standard L1 averages over all elements (B*T*4),
    # while our weighted loss averages per-box loss (sum over 4 coords)
    # then averages weighted frames, then averages batch. Results will differ.
    print(f"Standard L1 Loss (per element): {standard_l1.item()}")

    # --- Kalman Loss Example ---
    print("\n--- Kalman Loss Example ---")
    B, T = 2, 5 # Batch size 2, 5 frames
    kf_preds = torch.rand(B, T, 4)
    mamba_meas = torch.rand(B, T, 4)
    targets_bbox_kalman = torch.rand(B, T, 4)

    # Test with Smooth L1
    kalman_loss_smooth_l1 = KalmanLoss(kf_loss_weight=1.0, mamba_loss_weight=0.3, loss_type='smooth_l1')
    total_loss_smooth, components_smooth = kalman_loss_smooth_l1(kf_preds, mamba_meas, targets_bbox_kalman)
    print(f"Kalman Loss (Smooth L1): {total_loss_smooth.item():.4f}")
    print(f"Components (Smooth L1): {components_smooth}")

    # Test with L1
    kalman_loss_l1 = KalmanLoss(kf_loss_weight=1.0, mamba_loss_weight=0.3, loss_type='l1')
    total_loss_l1, components_l1 = kalman_loss_l1(kf_preds, mamba_meas, targets_bbox_kalman)
    print(f"Kalman Loss (L1): {total_loss_l1.item():.4f}")
    print(f"Components (L1): {components_l1}")

    # Test with MSE
    kalman_loss_mse = KalmanLoss(kf_loss_weight=1.0, mamba_loss_weight=0.3, loss_type='mse')
    total_loss_mse, components_mse = kalman_loss_mse(kf_preds, mamba_meas, targets_bbox_kalman)
    print(f"Kalman Loss (MSE): {total_loss_mse.item():.4f}")
    print(f"Components (MSE): {components_mse}")
