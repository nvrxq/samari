import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    LearningRateMonitor,
    EarlyStopping,
)
import logging
# Assuming your model, dataset, and loss are in these locations
from modeling.model import (
    TemporalFusionModule,
)
from dataset.sav_sync import SyncAVDataset
from modeling.loss import weighted_bce_loss
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
        last_frame_loss_weight=2.0,
    ):
        super().__init__()
        self.save_hyperparameters()  # Saves args like learning_rate to hparams

        self.model = TemporalFusionModule(model_config)
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.last_frame_loss_weight = last_frame_loss_weight

    def forward(self, frames, past_masks):
        # frames: (B, T, C, H, W)
        # past_masks: (B, T-1, 1, H, W)
        return self.model(frames, past_masks)

    def _common_step(self, batch, batch_idx, stage):
        frames = batch["frames"]
        past_masks = batch["masks"]
        target_masks = batch["label"]
        # frames: (B, T, C, H, W)
        # past_masks: (B, T-1, 1, H, W)
        # target_masks: (B, 1, 1, H, W) - Assuming target is only the last frame mask

        # Model predicts masks for T-1 frames + the target frame
        # Output shape: (B, T, 1, H, W)
        predicted_masks = self(frames, past_masks)

        # We need the ground truth for all predicted frames.
        # Assuming the input `past_masks` are the ground truth for the first T-1 frames
        # and `target_masks` is the ground truth for the T-th frame.
        # Concatenate them to match the prediction shape.
        all_target_masks = torch.cat(
            [past_masks, target_masks], dim=1
        )  # Shape: (B, T, 1, H, W)

        # Ensure shapes match before loss calculation
        if predicted_masks.shape != all_target_masks.shape:
            raise ValueError(
                f"Shape mismatch: Predicted {predicted_masks.shape} vs Target {all_target_masks.shape}"
            )

        loss = weighted_bce_loss(
            predicted_masks,
            all_target_masks,
            last_frame_weight=self.last_frame_loss_weight,
        )

        self.log(
            f"{stage}_loss",
            loss,
            on_step=(stage == "train"),
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )
        # Add other metrics if needed (e.g., IoU, Dice)
        # iou = calculate_iou(predicted_masks.sigmoid() > 0.5, all_target_masks > 0.5)
        # self.log(f'{stage}_iou', iou, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
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
    train_dataset = SyncAVDataset(
        sav_dir=dataset_config["train_data_dir"],
        frames_per_sample=dataset_config["frames_per_sample"],
    )
    val_dataset = SyncAVDataset(
        sav_dir=dataset_config["val_data_dir"],
        frames_per_sample=dataset_config["frames_per_sample"],
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
        last_frame_loss_weight=train_config["last_frame_loss_weight"],
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
        patience=10,  # Stop after 10 epochs with no improvement
        verbose=True,
        mode="min",
    )

    # --- Logger ---
    logger = TensorBoardLogger(train_config["log_dir"], name="temporal_fusion_model")
    util_logger.info(f"""Your training setup:
                     Epochs: {train_config["max_epochs"]}
                     Learning Rate: {train_config["learning_rate"]}
                     Weight Decay: {train_config["weight_decay"]}
                     Dataset len: {len(train_dataset)}
                     Model active params:{total_params / 1e6}M""")
    # --- Trainer ---
    trainer = pl.Trainer(
        accelerator=train_config["accelerator"],
        devices=train_config["devices"],
        strategy=(
            train_config["strategy"]
            if torch.cuda.device_count() > 1 and train_config["accelerator"] == "gpu"
            else "auto"
        ),  # Use DDP only if multiple GPUs
        max_epochs=train_config["max_epochs"],
        logger=logger,
        callbacks=[checkpoint_callback, lr_monitor, early_stop_callback],
        precision=train_config["precision"],
        log_every_n_steps=10,
        # Add other Trainer flags as needed (e.g., gradient_clip_val)
    )

    # --- Training ---
    print("Starting training...")
    trainer.fit(model, train_loader, val_loader)

    print("Training finished.")
    # Optionally run testing
    # print("Starting testing...")
    # trainer.test(model, test_loader) # Define test_loader similarly to val_loader
