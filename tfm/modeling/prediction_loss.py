import torch
import torch.nn as nn  # Added for KalmanLoss
import torch.nn.functional as F
import math  # Added for GIoU calculation


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


# --- BBox Utility Functions ---


def box_cxcywh_to_xyxy(boxes):
    """
    Convert boxes from [cx, cy, w, h] format to [x_min, y_min, x_max, y_max] format.
    Args:
        boxes (torch.Tensor): Boxes in [cx, cy, w, h] format. Shape (..., 4).
    Returns:
        torch.Tensor: Boxes in [x_min, y_min, x_max, y_max] format. Shape (..., 4).
    """
    cx, cy, w, h = boxes.unbind(-1)
    x1 = cx - 0.5 * w
    y1 = cy - 0.5 * h
    x2 = cx + 0.5 * w
    y2 = cy + 0.5 * h
    return torch.stack([x1, y1, x2, y2], dim=-1)


def box_xyxy_to_cxcywh(boxes):
    """
    Convert boxes from [x_min, y_min, x_max, y_max] format to [cx, cy, w, h] format.
    Args:
        boxes (torch.Tensor): Boxes in [x_min, y_min, x_max, y_max] format. Shape (..., 4).
    Returns:
        torch.Tensor: Boxes in [cx, cy, w, h] format. Shape (..., 4).
    """
    x1, y1, x2, y2 = boxes.unbind(-1)
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    w = x2 - x1
    h = y2 - y1
    return torch.stack([cx, cy, w, h], dim=-1)


def box_iou(boxes1_xyxy, boxes2_xyxy):
    """
    Calculate Intersection over Union (IoU) between two sets of boxes.
    Assumes boxes are in [x_min, y_min, x_max, y_max] format.

    Args:
        boxes1_xyxy (torch.Tensor): First set of boxes. Shape (N, 4).
        boxes2_xyxy (torch.Tensor): Second set of boxes. Shape (M, 4).

    Returns:
        torch.Tensor: IoU matrix of shape (N, M).
        torch.Tensor: Union matrix of shape (N, M).
    """
    area1 = (boxes1_xyxy[:, 2] - boxes1_xyxy[:, 0]) * (
        boxes1_xyxy[:, 3] - boxes1_xyxy[:, 1]
    )
    area2 = (boxes2_xyxy[:, 2] - boxes2_xyxy[:, 0]) * (
        boxes2_xyxy[:, 3] - boxes2_xyxy[:, 1]
    )

    # Calculate intersection coordinates
    lt = torch.max(boxes1_xyxy[:, None, :2], boxes2_xyxy[:, :2])  # (N, M, 2)
    rb = torch.min(boxes1_xyxy[:, None, 2:], boxes2_xyxy[:, 2:])  # (N, M, 2)

    wh = (rb - lt).clamp(min=0)  # (N, M, 2)
    intersection = wh[:, :, 0] * wh[:, :, 1]  # (N, M)

    union = area1[:, None] + area2 - intersection

    iou = intersection / (union + 1e-6)  # Add epsilon for stability

    return iou, union


# --- New Loss Function: Generalized IoU (GIoU) ---


def generalized_box_iou_loss(preds_cxcywh, targets_cxcywh, reduction="mean"):
    """
    Generalized IoU loss (https://giou.stanford.edu/).
    Combines IoU with a penalty for non-overlapping boxes.
    Assumes inputs are in [cx, cy, w, h] format (normalized).

    Args:
        preds_cxcywh (torch.Tensor): Predicted boxes (B, T, 4) or (N, 4).
        targets_cxcywh (torch.Tensor): Target boxes (B, T, 4) or (N, 4).
        reduction (str): 'mean', 'sum', or 'none'.

    Returns:
        torch.Tensor: Calculated GIoU loss.
    """
    # Ensure inputs are float
    preds_cxcywh = preds_cxcywh.float()
    targets_cxcywh = targets_cxcywh.float()

    # Flatten batch and time dimensions if present
    original_shape = preds_cxcywh.shape
    if len(original_shape) > 2:
        preds_cxcywh = preds_cxcywh.view(-1, 4)
        targets_cxcywh = targets_cxcywh.view(-1, 4)
    N = preds_cxcywh.shape[0]
    if N == 0:
        return torch.tensor(
            0.0, device=preds_cxcywh.device, requires_grad=True
        )  # Handle empty input

    # Convert to [x1, y1, x2, y2]
    preds_xyxy = box_cxcywh_to_xyxy(preds_cxcywh)
    targets_xyxy = box_cxcywh_to_xyxy(targets_cxcywh)

    # Calculate IoU and Union (we only need diagonal elements for loss)
    # Ensure boxes are valid (x2 > x1, y2 > y1) - clamp width/height in cxcywh?
    # Clamping might hide issues, better to ensure valid inputs or handle invalid boxes
    # For simplicity here, assume valid inputs or rely on box_iou handling

    # Calculate IoU for corresponding pairs
    iou, union = box_iou(preds_xyxy, targets_xyxy)
    iou = torch.diag(iou)  # Get IoU for matching pairs (preds[i] vs targets[i])

    # Calculate enclosing box C coordinates
    x1_c = torch.min(preds_xyxy[:, 0], targets_xyxy[:, 0])
    y1_c = torch.min(preds_xyxy[:, 1], targets_xyxy[:, 1])
    x2_c = torch.max(preds_xyxy[:, 2], targets_xyxy[:, 2])
    y2_c = torch.max(preds_xyxy[:, 3], targets_xyxy[:, 3])

    # Calculate area of enclosing box C
    area_c = (x2_c - x1_c) * (y2_c - y1_c)

    # Calculate GIoU
    giou = iou - (area_c - torch.diag(union)) / (area_c + 1e-6)

    # GIoU loss is 1 - GIoU
    loss = 1.0 - giou

    # Apply reduction
    if reduction == "mean":
        loss = loss.mean()
    elif reduction == "sum":
        loss = loss.sum()
    elif reduction != "none":
        raise ValueError(f"Unsupported reduction: {reduction}")

    # Reshape back if needed (though usually mean/sum is used)
    # if reduction == 'none' and len(original_shape) > 2:
    #     loss = loss.view(original_shape[:-1])

    return loss


# +++ Kalman Loss Function (Moved from model.py) +++
class KalmanLoss(nn.Module):
    """
    Combines losses for Kalman Filter predictions and Mamba measurements
    against target bounding boxes. Supports L1, Smooth L1, MSE, and GIoU.
    Handles padding using an attention mask.
    """

    def __init__(
        self, kf_loss_weight=1.0, mamba_loss_weight=0.3, loss_type="smooth_l1"
    ):
        super().__init__()
        self.kf_loss_weight = kf_loss_weight
        self.mamba_loss_weight = mamba_loss_weight
        self.loss_type = loss_type.lower()

        # Note: Reduction is handled manually based on the mask
        if self.loss_type == "l1":
            self.loss_fn = F.l1_loss
        elif self.loss_type == "smooth_l1":
            self.loss_fn = F.smooth_l1_loss
        elif self.loss_type == "mse":
            self.loss_fn = F.mse_loss
        elif self.loss_type == "giou":
            # GIoU loss expects cxcywh format and handles reduction internally
            # We will pass only the valid elements to it.
            self.loss_fn = generalized_box_iou_loss
        else:
            raise ValueError(f"Unsupported loss_type: {loss_type}")

    def forward(
        self, kf_predictions, mamba_measurements, target_bboxes, attention_mask=None
    ):
        """
        Calculate the combined loss, considering the attention mask for padding.

        Args:
            kf_predictions (torch.Tensor): Final predictions from Kalman Filter (B, T, 4). [cx, cy, w, h]
            mamba_measurements (torch.Tensor): Intermediate measurements from Mamba head (B, T, 4). [cx, cy, w, h]
            target_bboxes (torch.Tensor): Ground truth bounding boxes (B, T, 4). [cx, cy, w, h]
            attention_mask (torch.Tensor, optional): Boolean mask indicating valid time steps (B, T).
                                                     True for valid, False for padding. Defaults to None (no padding).

        Returns:
            torch.Tensor: The total calculated loss.
            dict: Dictionary containing individual loss components.
        """
        # Ensure tensors are float
        target_bboxes = target_bboxes.float()
        kf_predictions = kf_predictions.float()
        mamba_measurements = mamba_measurements.float()

        B, T, C = target_bboxes.shape

        if attention_mask is None:
            # If no mask provided, assume all elements are valid
            attention_mask = torch.ones(
                B, T, dtype=torch.bool, device=target_bboxes.device
            )

        # Ensure mask has the correct shape (B, T)
        if attention_mask.shape != (B, T):
            raise ValueError(
                f"Attention mask shape {attention_mask.shape} does not match target shape ({B}, {T})"
            )

        # Count the number of valid (non-padded) elements across the batch
        # This is crucial for correct averaging of the loss
        num_valid_elements = attention_mask.sum()

        if num_valid_elements == 0:
            # Avoid division by zero if the batch contains only padding (should not happen with min_seq_len > 0)
            loss_kf = torch.tensor(0.0, device=target_bboxes.device, requires_grad=True)
            loss_mamba = torch.tensor(
                0.0, device=target_bboxes.device, requires_grad=True
            )
        else:
            # Expand mask shape to match bbox shape (B, T, 4) for element-wise selection/multiplication
            mask_expanded = attention_mask.unsqueeze(-1).expand_as(
                target_bboxes
            )  # (B, T, 4)

            if self.loss_type == "giou":
                # For GIoU, we need to select the valid boxes *before* passing them to the loss function
                # Flatten predictions and targets, then select based on the mask
                kf_preds_flat = kf_predictions[attention_mask]  # (N_valid, 4)
                mamba_meas_flat = mamba_measurements[attention_mask]  # (N_valid, 4)
                targets_flat = target_bboxes[attention_mask]  # (N_valid, 4)

                # GIoU loss function handles reduction='mean' internally over the valid elements provided
                loss_kf = self.loss_fn(kf_preds_flat, targets_flat, reduction="mean")
                loss_mamba = self.loss_fn(
                    mamba_meas_flat, targets_flat, reduction="mean"
                )
            else:
                # For L1, SmoothL1, MSE, calculate loss per element and then average over valid elements
                # Calculate loss with reduction='none' to get per-element loss
                loss_kf_per_element = self.loss_fn(
                    kf_predictions, target_bboxes, reduction="none"
                )
                loss_mamba_per_element = self.loss_fn(
                    mamba_measurements, target_bboxes, reduction="none"
                )

                # Apply mask: Zero out loss for padded elements
                # Note: Multiplying by mask_expanded (which is 0/1) works here
                masked_loss_kf = loss_kf_per_element * mask_expanded
                masked_loss_mamba = loss_mamba_per_element * mask_expanded

                # Sum the loss over all elements (valid ones contribute, padded ones are zero)
                # Then divide by the number of *valid* elements for the correct mean
                loss_kf = masked_loss_kf.sum() / num_valid_elements
                loss_mamba = masked_loss_mamba.sum() / num_valid_elements

        # Combine losses with weights
        total_loss = self.kf_loss_weight * loss_kf + self.mamba_loss_weight * loss_mamba

        loss_dict = {
            "total_loss": total_loss,
            f"loss_kf_{self.loss_type}": loss_kf,
            f"loss_mamba_{self.loss_type}": loss_mamba,
        }
        # Keep old keys for potential backward compatibility if needed elsewhere
        loss_dict["loss_kf"] = loss_kf
        loss_dict["loss_mamba"] = loss_mamba

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

    # --- GIoU Loss Example ---
    print("\n--- GIoU Loss Example ---")
    B, T = 2, 5
    # Use realistic cx, cy, w, h values (normalized)
    preds_bbox_cxcywh = torch.rand(B, T, 4) * 0.8 + 0.1  # Avoid edges initially
    targets_bbox_cxcywh = torch.rand(B, T, 4) * 0.8 + 0.1
    # Ensure width/height are positive
    preds_bbox_cxcywh[..., 2:] = preds_bbox_cxcywh[..., 2:].clamp(min=0.01)
    targets_bbox_cxcywh[..., 2:] = targets_bbox_cxcywh[..., 2:].clamp(min=0.01)

    loss_giou = generalized_box_iou_loss(preds_bbox_cxcywh, targets_bbox_cxcywh)
    print(f"Calculated GIoU Loss: {loss_giou.item():.4f}")

    # Test GIoU with non-overlapping boxes
    preds_nonoverlap = torch.tensor([[[0.1, 0.1, 0.1, 0.1]]])  # Top-left
    targets_nonoverlap = torch.tensor([[[0.9, 0.9, 0.1, 0.1]]])  # Bottom-right
    loss_giou_nonoverlap = generalized_box_iou_loss(
        preds_nonoverlap, targets_nonoverlap
    )
    print(
        f"GIoU Loss (non-overlapping): {loss_giou_nonoverlap.item():.4f}"
    )  # Should be > 1

    # --- Kalman Loss with Masking Example ---
    print("\n--- Kalman Loss with Masking Example ---")
    B, T = 2, 5
    kf_preds = torch.rand(B, T, 4)
    mamba_meas = torch.rand(B, T, 4)
    targets_bbox_kalman = torch.rand(B, T, 4)
    # Ensure positive w/h
    kf_preds[..., 2:] = kf_preds[..., 2:].clamp(min=0.01)
    mamba_meas[..., 2:] = mamba_meas[..., 2:].clamp(min=0.01)
    targets_bbox_kalman[..., 2:] = targets_bbox_kalman[..., 2:].clamp(min=0.01)

    # Create a sample mask (e.g., first sequence has length 3, second has length 5)
    mask = torch.zeros(B, T, dtype=torch.bool)
    mask[0, :3] = True
    mask[1, :5] = True
    print("Sample Attention Mask:\n", mask)
    num_valid = mask.sum()
    print("Number of valid elements:", num_valid.item())

    # Test with GIoU
    kalman_loss_giou_masked = KalmanLoss(
        kf_loss_weight=1.0, mamba_loss_weight=0.5, loss_type="giou"
    )
    total_loss_giou_m, components_giou_m = kalman_loss_giou_masked(
        kf_preds, mamba_meas, targets_bbox_kalman, attention_mask=mask
    )
    print(f"\nKalman Loss (GIoU, Masked): {total_loss_giou_m.item():.4f}")
    print(f"Components (GIoU, Masked): {components_giou_m}")

    # Test with Smooth L1
    kalman_loss_smooth_l1_masked = KalmanLoss(
        kf_loss_weight=1.0, mamba_loss_weight=0.5, loss_type="smooth_l1"
    )
    total_loss_smooth_m, components_smooth_m = kalman_loss_smooth_l1_masked(
        kf_preds, mamba_meas, targets_bbox_kalman, attention_mask=mask
    )
    print(f"\nKalman Loss (Smooth L1, Masked): {total_loss_smooth_m.item():.4f}")
    print(f"Components (Smooth L1, Masked): {components_smooth_m}")

    # Compare Smooth L1 without mask (should differ if padding exists)
    total_loss_smooth_nomask, _ = kalman_loss_smooth_l1_masked(
        kf_preds,
        mamba_meas,
        targets_bbox_kalman,
        attention_mask=None,  # Or don't pass mask
    )
    print(f"\nKalman Loss (Smooth L1, No Mask): {total_loss_smooth_nomask.item():.4f}")
