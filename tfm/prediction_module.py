

class TemporalFusionTrainer(pl.LightningModule):
    def __init__(
        self,
        model_config,
        learning_rate=1e-4,
        weight_decay=1e-5,
        loss_type: str = "smooth_l1",
        kf_loss_weight: float = 1.0,
        mamba_loss_weight: float = 0.2,
    ):
        super().__init__()
        self.save_hyperparameters()  # Saves args like learning_rate to hparams

        self.model = TemporalFusionModule(model_config)
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.loss_fn = KalmanLoss(
            kf_loss_weight=kf_loss_weight,
            mamba_loss_weight=mamba_loss_weight,
            loss_type=loss_type,
        )

    def forward(self, frames, target_bboxes):
        # Pass arguments directly to the underlying model's forward
        # frames: (B, T, C, H, W)
        # target_bboxes: (B, T, 4)
        return self.model(frames, target_bboxes)

    def _calculate_iou_metric(self, preds_cxcywh, targets_cxcywh):
        """
        Helper to calculate IoU for metrics.
        DEPRECATED: IoU calculation moved to _common_step to handle padding.
        Kept for potential future use or reference.
        """
        # Select last frame predictions and targets
        last_frame_pred = preds_cxcywh[:, -1, :]  # (B, 4)
        last_frame_target = targets_cxcywh[:, -1, :]  # (B, 4)

        # Convert to xyxy format for IoU calculation
        last_frame_pred_xyxy = box_cxcywh_to_xyxy(last_frame_pred)
        last_frame_target_xyxy = box_cxcywh_to_xyxy(last_frame_target)

        # Calculate IoU (returns diagonal of the IoU matrix if inputs are Bx4)
        # Ensure inputs are flattened for pairwise IoU if needed, but here B==B
        iou, _ = box_iou(last_frame_pred_xyxy, last_frame_target_xyxy)
        # box_iou returns (N, M), we need the diagonal elements where N=M=B
        diag_iou = torch.diag(iou)  # Shape (B,)

        # Return mean IoU over the batch
        return diag_iou.mean()

    def _common_step(self, batch, batch_idx, stage):
        frames = batch["frames"]  # Shape: (B, T_max, C, H, W)
        target_bboxes = batch[
            "target_bboxes"
        ]  # Shape: (B, T_max, 4) - Normalized cxcywh
        attention_mask = batch.get(
            "attention_mask"
        )  # Shape: (B, T_max) boolean or None

        # Model predicts KF outputs and Mamba measurements for all T_max frames
        # Output shapes: (B, T_max, 4), (B, T_max, 4)
        kf_predictions, mamba_measurements = self(frames, target_bboxes)

        # Ensure shapes match before loss calculation (checking T_max)
        if kf_predictions.shape != target_bboxes.shape:
            raise ValueError(
                f"Shape mismatch KF: Predicted {kf_predictions.shape} vs Target {target_bboxes.shape}"
            )
        if mamba_measurements.shape != target_bboxes.shape:
            raise ValueError(
                f"Shape mismatch Mamba: Predicted {mamba_measurements.shape} vs Target {target_bboxes.shape}"
            )
        if (
            kf_predictions.shape[1] != frames.shape[1]
        ):  # Check T_max dimension consistency
            raise ValueError(
                f"Temporal dimension mismatch: Predicted T={kf_predictions.shape[1]} vs Input Frames T={frames.shape[1]}"
            )

        # Calculate loss using KalmanLoss, passing the mask
        # Ensure your KalmanLoss implementation accepts and uses the attention_mask
        loss, loss_stat = self.loss_fn(
            kf_predictions,
            mamba_measurements,
            target_bboxes,
            attention_mask=attention_mask,  # Pass the mask here
        )

        # Check for NaN loss
        if torch.isnan(loss):
            util_logger.error(f"NaN loss detected in {stage} step!")
            # Potentially log inputs/outputs or raise an error for debugging
            # For now, just log it. Consider stopping training if it persists.
            # raise ValueError("NaN loss detected")

        # Log individual loss components from KalmanLoss dictionary
        self.log(
            f"{stage}_loss",
            loss_stat["total_loss"],
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )
        kf_loss_key = f"loss_kf_{self.hparams.loss_type}"
        mamba_loss_key = f"loss_mamba_{self.hparams.loss_type}"
        if kf_loss_key in loss_stat:
            self.log(
                f"{stage}_loss_kf",
                loss_stat[kf_loss_key],
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                logger=True,
                sync_dist=True,
            )
        if mamba_loss_key in loss_stat:
            self.log(
                f"{stage}_loss_mamba",
                loss_stat[mamba_loss_key],
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                logger=True,
                sync_dist=True,
            )

        # Calculate and log IoU for the last *actual* frame using the attention mask
        if attention_mask is not None and kf_predictions.shape[0] > 0:
            with torch.no_grad():  # No need to track gradients for metrics
                # Get sequence lengths from the mask (sum of True values)
                seq_lengths = attention_mask.sum(dim=1)  # Shape: (B,)
                last_frame_indices = seq_lengths - 1  # Indices are 0-based

                # Handle sequences with length 0 if they somehow occur (shouldn't with min_seq_len=2)
                valid_indices = last_frame_indices >= 0
                if valid_indices.any():
                    batch_indices = torch.arange(
                        kf_predictions.shape[0], device=kf_predictions.device
                    )[valid_indices]
                    last_indices_valid = last_frame_indices[valid_indices]

                    # Gather the predictions and targets for the last valid frame of each sequence
                    last_frame_pred = kf_predictions[
                        batch_indices, last_indices_valid, :
                    ]  # (B_valid, 4)
                    last_frame_target = target_bboxes[
                        batch_indices, last_indices_valid, :
                    ]  # (B_valid, 4)

                    # Convert cxcywh to xyxy format for IoU calculation
                    last_frame_pred_xyxy = box_cxcywh_to_xyxy(last_frame_pred)
                    last_frame_target_xyxy = box_cxcywh_to_xyxy(last_frame_target)

                    # Calculate IoU
                    iou, _ = box_iou(last_frame_pred_xyxy, last_frame_target_xyxy)
                    diag_iou = torch.diag(iou)  # Shape (B_valid,)
                    last_frame_iou_mean = (
                        diag_iou.mean()
                    )  # Mean IoU over valid sequences in the batch
                else:
                    # No valid sequences in the batch to calculate IoU for
                    last_frame_iou_mean = torch.tensor(
                        0.0, device=kf_predictions.device
                    )

            self.log(
                f"{stage}_last_iou",
                last_frame_iou_mean,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                logger=True,
                sync_dist=True,
            )
        # Fallback if mask isn't provided (should not happen with collate_fn)
        elif kf_predictions.shape[0] > 0:
            util_logger.warning(
                f"Attention mask not found in {stage} step, calculating IoU on last padded frame."
            )
            with torch.no_grad():
                # This uses the potentially padded last frame, less accurate
                last_frame_iou = self._calculate_iou_metric(
                    kf_predictions, target_bboxes
                )
            self.log(
                f"{stage}_last_iou",
                last_frame_iou,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                logger=True,
                sync_dist=True,
            )

        return loss

    def training_step(self, batch, batch_idx):
        return self._common_step(batch, batch_idx, "train")

    def validation_step(self, batch, batch_idx):
        return self._common_step(batch, batch_idx, "val")

    def test_step(self, batch, batch_idx):
        return self._common_step(batch, batch_idx, "test")

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
        )
        # Optional: Add a learning rate scheduler
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.1, patience=5, verbose=True
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",  # Metric to monitor for scheduler
                "interval": "epoch",
                "frequency": 1,
            },
        }




# --- Custom Collate Function ---
def collate_fn(batch):
    """
    Pads sequences within a batch to the maximum sequence length.
    Args:
        batch: A list of dictionaries from the dataset, e.g.,
               [{'frames': tensor(T1, C, H, W), 'target_bboxes': tensor(T1, 4)}, ...]
    Returns:
        A dictionary containing batched & padded tensors:
        {
            'frames': tensor(B, T_max, C, H, W),
            'target_bboxes': tensor(B, T_max, 4), # cxcywh format
            'attention_mask': tensor(B, T_max) # boolean mask (True for real data)
        }
    """
    # Separate frames and target_bboxes
    frames_list = [item["frames"] for item in batch]  # List of (T_i, C, H, W) tensors
    bboxes_list = [item["target_bboxes"] for item in batch]  # List of (T_i, 4) tensors

    # Get sequence lengths for each item
    seq_lengths = [len(f) for f in frames_list]
    if not seq_lengths:  # Handle empty batch
        return {
            "frames": torch.empty(0),
            "target_bboxes": torch.empty(0),
            "attention_mask": torch.empty(0),
        }
    max_len = max(seq_lengths)

    # Pad sequences
    # pad_sequence expects batch_first=False by default, input shape (T, *)
    # Our frames are (T, C, H, W), bboxes are (T, 4) which is correct
    # We need padding_value=0.0 for tensors
    # Set batch_first=True for (B, T, ...) output shape
    padded_frames = pad_sequence(
        frames_list, batch_first=True, padding_value=0.0
    )  # (B, max_len, C, H, W)
    padded_bboxes = pad_sequence(
        bboxes_list, batch_first=True, padding_value=0.0
    )  # (B, max_len, 4)

    # Create attention mask (True for real data, False for padding)
    attention_mask = torch.zeros(len(batch), max_len, dtype=torch.bool)
    for i, length in enumerate(seq_lengths):
        attention_mask[i, :length] = True  # (B, max_len)

    return {
        "frames": padded_frames,
        "target_bboxes": padded_bboxes,
        "attention_mask": attention_mask,  # Add mask to the batch output
    }
