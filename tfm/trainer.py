import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    LearningRateMonitor,
    EarlyStopping,
)
import logging
import os

# Assuming your model, dataset, and loss are in these locations
from modeling.model import (
    TemporalFusionModule,
)
from dataset.lasot_dataset import LaSotDataset
from modeling.loss import KalmanLoss
import argparse


util_logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="tfm/configs/train_tiny.yaml")
    return parser.parse_args()


class TemporalFusionTrainer(pl.LightningModule):
    def __init__(
        self,
        model_config,
        learning_rate=1e-4,
        weight_decay=1e-5,
        loss_type: str = 'smooth_l1',
        kf_loss_weight: float = 1.0,
        mamba_loss_weight: float = 0.2
    ):
        super().__init__()
        self.save_hyperparameters()  # Saves args like learning_rate to hparams

        self.model = TemporalFusionModule(model_config)
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.loss_fn = KalmanLoss(
            kf_loss_weight=kf_loss_weight,
            mamba_loss_weight=mamba_loss_weight,
            loss_type=loss_type
        )

    def forward(self, frames, target_bboxes):
        # Pass arguments directly to the underlying model's forward
        # frames: (B, T, C, H, W)
        # target_bboxes: (B, T, 4)
        return self.model(frames, target_bboxes)

    def _common_step(self, batch, batch_idx, stage):
        frames = batch["frames"]              # Shape: (B, T, C, H, W)
        target_bboxes = batch["target_bboxes"] # Shape: (B, T, 4) - Normalized

        # Model predicts KF outputs and Mamba measurements for all T frames
        # Output shapes: (B, T, 4), (B, T, 4)
        kf_predictions, mamba_measurements = self(frames, target_bboxes)

        # Ensure shapes match before loss calculation
        if kf_predictions.shape != target_bboxes.shape:
            raise ValueError(
                f"Shape mismatch KF: Predicted {kf_predictions.shape} vs Target {target_bboxes.shape}"
            )
        if mamba_measurements.shape != target_bboxes.shape:
             raise ValueError(
                 f"Shape mismatch Mamba: Predicted {mamba_measurements.shape} vs Target {target_bboxes.shape}"
             )
        if kf_predictions.shape[1] != frames.shape[1]: # Check T dimension consistency
             raise ValueError(
                f"Temporal dimension mismatch: Predicted T={kf_predictions.shape[1]} vs Input Frames T={frames.shape[1]}"
            )

        # Calculate loss using KalmanLoss
        loss, loss_stat = self.loss_fn(
            kf_predictions,
            mamba_measurements,
            target_bboxes,
            # last_frame_weight is handled inside KalmanLoss if needed,
            # but current KalmanLoss doesn't use it.
        )

        # Check for NaN loss
        if torch.isnan(loss):
            util_logger.error(f"NaN loss detected in {stage} step!")
            # Potentially log inputs/outputs or raise an error for debugging
            # For now, just log it. Consider stopping training if it persists.
            # raise ValueError("NaN loss detected")

        # Log individual loss components from KalmanLoss dictionary
        self.log(f"{stage}_loss", loss_stat['total_loss'], on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log(f"{stage}_loss_kf", loss_stat['loss_kf'], on_step=False, on_epoch=True, prog_bar=False, logger=True, sync_dist=True)
        self.log(f"{stage}_loss_mamba", loss_stat['loss_mamba'], on_step=False, on_epoch=True, prog_bar=False, logger=True, sync_dist=True)

        # Add other metrics if needed (e.g., IoU for the last frame)
        # last_frame_pred = kf_predictions[:, -1, :]
        # last_frame_target = target_bboxes[:, -1, :]
        # iou = calculate_iou(last_frame_pred, last_frame_target) # Assuming calculate_iou exists
        # self.log(f'{stage}_last_iou', iou, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
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


# --- Training Script ---
if __name__ == "__main__":
    import yaml
    import wandb

    args = parse_args()
    config = yaml.safe_load(open(args.config))
    dataset_config = config["dataset"]
    train_config = config["train"]
    model_config = config["model"]

    # --- DataLoaders ---
    # Note: AsyncAVDataset might need adjustments if used with multiple workers in DDP.
    # Consider using a standard PyTorch Dataset or ensuring thread/process safety.
    # For simplicity, using standard loading here. If AsyncAVDataset causes issues,
    # replace it with a simpler synchronous version for initial testing.
    train_dataset = LaSotDataset(
        root_dir=dataset_config["train_data_dir"],
        steps_per_epoch=train_config.get("steps_per_epoch", 1000), # Use config value or default
        max_frames=dataset_config["max_frames"],
        # transform=transform # Add transform if needed
    )
    val_dataset = LaSotDataset(
        root_dir=dataset_config["val_data_dir"], # Use separate validation dir if available
        steps_per_epoch=train_config.get("val_steps_per_epoch", 200), # Use config value or default
        max_frames=dataset_config["max_frames"],
        # transform=transform # Add transform if needed
    )

    # Important: shuffle=True for training, False for validation
    # persistent_workers=True can speed up loading after the first epoch
    train_loader = DataLoader(
        train_dataset,
        batch_size=dataset_config["batch_size"],
        shuffle=True,
        num_workers=dataset_config["num_workers"],
        pin_memory=True,
        persistent_workers=True if dataset_config["num_workers"] > 0 else False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=dataset_config["batch_size"],
        shuffle=False,
        num_workers=dataset_config["num_workers"],
        pin_memory=True,
        persistent_workers=True if dataset_config["num_workers"] > 0 else False,
    )

    # --- Model ---
    model = TemporalFusionTrainer(
        model_config=model_config,
        learning_rate=train_config["learning_rate"],
        weight_decay=train_config["weight_decay"],
        loss_type=train_config.get("loss_type", "smooth_l1"),
        kf_loss_weight=train_config.get("kf_loss_weight", 1.0),
        mamba_loss_weight=train_config.get("mamba_loss_weight", 0.2),
    )
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # --- Callbacks ---
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        dirpath=train_config["log_dir"],
        filename="temporal-fusion-{epoch:02d}-{val_loss:.2f}",
        save_top_k=3,
        mode="min",
    )
    lr_monitor = LearningRateMonitor(logging_interval="epoch")
    early_stop_callback = EarlyStopping(
        monitor="val_loss",
        patience=train_config.get("early_stopping_patience", 10),
        verbose=True,
        mode="min",
    )

    # --- Logger ---
    wandb_entity = os.getenv("WANDB_ENTITY")
    wandb_project = os.getenv("WANDB_PROJECT", "temporal-fusion-model")

    if not wandb_entity:
        util_logger.warning("WANDB_ENTITY environment variable not set. Using default WandB behavior.")

    logger = WandbLogger(
        name=train_config.get("run_name", None),
        project=wandb_project,
        entity=wandb_entity,
        log_model="all",
        config=config,
        save_dir=train_config["log_dir"]
    )

    util_logger.info(
        f"""Your training setup:
                     Epochs: {train_config["max_epochs"]}
                     Learning Rate: {train_config["learning_rate"]}
                     Weight Decay: {train_config["weight_decay"]}
                     Dataset len: {len(train_dataset)}
                     Model active params:{total_params / 1e6}M
                     Logging to WANDB project: {wandb_project}"""
                     + (f" entity: {wandb_entity}" if wandb_entity else "")
    )
    # --- Trainer ---
    trainer = pl.Trainer(
        accelerator=train_config["accelerator"],
        devices=train_config["devices"],
        strategy=(
            train_config["strategy"]
            if torch.cuda.device_count() > 1 and train_config["accelerator"] == "gpu"
            else "auto"
        ),
        max_epochs=train_config["max_epochs"],
        logger=logger,
        callbacks=[checkpoint_callback, lr_monitor, early_stop_callback],
        precision=train_config["precision"],
        log_every_n_steps=train_config.get("log_every_n_steps", 10),
        # Add other Trainer flags as needed (e.g., gradient_clip_val)
    )

    # --- Training ---
    print("Starting training with WandB logging...")
    try:
        trainer.fit(model, train_loader, val_loader)
    finally:
        wandb.finish()

    print("Training finished.")
    # Optionally run testing
    # print("Starting testing...")
    # trainer.test(model, test_loader) # Define test_loader similarly to val_loader
