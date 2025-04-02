import torch
import torch.nn.functional as F


def weighted_bce_loss(predicted_masks, target_masks, last_frame_weight=2.0):
    """
    Calculates Binary Cross Entropy loss with a higher weight for the last frame.

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


# Example Usage (can be removed or kept for testing)
if __name__ == "__main__":
    B, N, H, W = 2, 5, 1, 1  # Batch size 2, 5 frames, 64x64 resolution
    # Example predictions (logits before sigmoid are often used with BCEWithLogitsLoss,
    # but here we assume predictions are probabilities after sigmoid)
    preds = torch.rand(B, N, 1, H, W)
    # Example targets (binary masks)
    frames_mask = torch.rand(B, N - 1, 1, H, W)  # Mask from past frames[1: ]
    target_mask = torch.rand(B, 1, 1, H, W)  # Label Mask

    targets = torch.cat((frames_mask, target_mask), 1)
    assert targets.size(1) == N, "Mismatch"
    loss = weighted_bce_loss(preds, targets, last_frame_weight=3.0)
    print(f"Calculated Weighted BCE Loss: {loss.item()}")

    # Test with N=1
    preds_single = torch.rand(B, 1, 1, H, W)
    targets_single = torch.randint(0, 2, (B, 1, 1, H, W)).float()
    loss_single = weighted_bce_loss(preds_single, targets_single, last_frame_weight=3.0)
    print(f"Calculated Weighted BCE Loss (N=1): {loss_single.item()}")

    # Compare with standard BCE
    standard_bce = F.binary_cross_entropy(preds, targets)
    print(f"Standard BCE Loss: {standard_bce.item()}")
