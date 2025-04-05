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
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.nn.utils.rnn import pad_sequence

from dataset.lasot_dataset import LaSotDataset
from correction_module import CorrectionModule, lasot_collate_fn
import argparse


util_logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="tfm/configs/train_tiny.yaml")
    return parser.parse_args()


# --- Training Script ---
if __name__ == "__main__":
    import yaml
    import wandb

    args = parse_args()
    config = yaml.safe_load(open(args.config))
    dataset_config = config["dataset"]
    train_config = config["train"]
    model_config = config["model"]

    image_size = dataset_config.get("image_size", [256, 256])  # Default H, W
    img_mean = dataset_config.get("mean", [0.485, 0.456, 0.406])  # ImageNet mean
    img_std = dataset_config.get("std", [0.229, 0.224, 0.225])  # ImageNet std
    train_augmentations = A.Compose(
        [
            A.Resize(height=image_size[0], width=image_size[1]),
            A.HorizontalFlip(p=0.5),
            A.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1, p=0.5),
            A.RandomBrightnessContrast(p=0.2),
            A.GaussianBlur(p=0.3),
            A.Normalize(mean=img_mean, std=img_std),
            ToTensorV2(),
        ],
        bbox_params=A.BboxParams(
            format="pascal_voc", min_visibility=0.1, label_fields=["labels"]
        ),
    )

    base_transform = A.Compose(
        [
            A.Resize(height=image_size[0], width=image_size[1]),
            A.Normalize(mean=img_mean, std=img_std),
            ToTensorV2(),  # Converts image to PyTorch tensor (C, H, W)
        ],
        bbox_params=A.BboxParams(
            format="pascal_voc", min_visibility=0.1, label_fields=["labels"]
        ),
    )

        # --- DataLoaders ---
    train_dataset = LaSotDataset(
        root_dir=dataset_config["train_data_dir"],
        steps_per_epoch=train_config.get("steps_per_epoch", 1000),
        max_frames=dataset_config["max_frames"],
        transform=train_augmentations,  # Pass augmentations
        mode=train_config["task_type"],
    )
    val_dataset = LaSotDataset(
        root_dir=dataset_config["val_data_dir"],
        steps_per_epoch=train_config.get("val_steps_per_epoch", 200),
        max_frames=dataset_config["max_frames"],
        transform=base_transform,
        mode=train_config["task_type"],
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=dataset_config["batch_size"],
        shuffle=True,
        num_workers=dataset_config["num_workers"],
        pin_memory=True,
        persistent_workers=True if dataset_config["num_workers"] > 0 else False,
        collate_fn=lasot_collate_fn,  # Use custom collate function
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=dataset_config["batch_size"],
        shuffle=False,
        num_workers=dataset_config["num_workers"],
        pin_memory=True,
        persistent_workers=True if dataset_config["num_workers"] > 0 else False,
        collate_fn=lasot_collate_fn,  # Use custom collate function
    )

    # --- Model ---
    if train_config["task_type"] == "correction":
        model = CorrectionModule(model_config=model_config, train_config=train_config)
    else:
        model = None
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # --- Callbacks ---
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        dirpath=train_config["log_dir"],
        filename="correction-{epoch:02d}-{val_loss:.2f}",
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
    wandb_project = os.getenv("WANDB_PROJECT", "correction-model")

    if not wandb_entity:
        util_logger.warning(
            "WANDB_ENTITY environment variable not set. Using default WandB behavior."
        )

    logger = WandbLogger(
        name=train_config.get("run_name", None),
        project=wandb_project,
        entity=wandb_entity,
        log_model="all",
        config=config,
        save_dir=train_config["log_dir"],
    )

    util_logger.info(
        f"""Your training setup:
                     Epochs: {train_config["max_epochs"]}
                     Learning Rate: {train_config["learning_rate"]}
                     Weight Decay: {train_config["weight_decay"]}
                     Dataset len: {len(train_dataset)}
                     Model active params:{total_params / 1e6}M
                     Logging to WANDB project: {wandb_project}
                     Task type: {train_config["task_type"]}
"""
        + (f" entity: {wandb_entity}" if wandb_entity else "")
    )
    # --- Trainer ---
    trainer = pl.Trainer(
        accelerator=train_config["accelerator"],
        devices=train_config["devices"],
        strategy=train_config.get("strategy", "auto"),
        max_epochs=train_config["max_epochs"],
        logger=logger,
        callbacks=[checkpoint_callback, lr_monitor, early_stop_callback],
        precision="16-mixed",  # Use mixed precision training with float16
        log_every_n_steps=train_config.get("log_every_n_steps", 10),
        # Add other Trainer flags as needed (e.g., gradient_clip_val)
    )

    # --- Training ---
    print("Starting training with WandB logging...")
    try:
        trainer.fit(model, train_loader, val_loader)
    finally:
        if wandb.run is not None:  # Check if wandb run was initialized
            wandb.finish()

    print("Training finished.")
    # Optionally run testing
    # print("Starting testing...")
    # trainer.test(model, test_loader) # Define test_loader similarly to val_loader
