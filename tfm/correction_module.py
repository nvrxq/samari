import torch
import pytorch_lightning as pl
from tfm.modeling.correction_model import STMTKF
from tfm.modeling.correction_loss import STMTKFLoss
from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger
import matplotlib.pyplot as plt


class CorrectionModule(pl.LightningModule):
    def __init__(self, model_config, train_config):
        super().__init__()
        self.save_hyperparameters()
        self.lr_scheduler = train_config.get("lr_scheduler", "cosine")
        self.model = STMTKF(config=model_config, device=self.device)

        self.learning_rate = train_config.get("learning_rate", 1e-4)
        self.weight_decay = train_config.get("weight_decay", 1e-5)

        self.loss_type = train_config.get("loss_type", "bbox")
        self.lambda_kf = train_config.get("lambda_kf", 0.1)
        self.lambda_motion = train_config.get("lambda_motion", 0.4)
        self.lambda_struct = train_config.get("lambda_struct", 0.3)
        self.lambda_temp = train_config.get("lambda_temp", 0.5)
        self.adaptive_weighting = train_config.get("adaptive_weighting", True)
        self.focal_gamma = train_config.get("focal_gamma", 2.0)
        self.input_type = train_config.get("input_type", "bbox")

        self.loss_fn = STMTKFLoss(
            input_type=self.loss_type,
            lambda_kf=self.lambda_kf,
            lambda_motion=self.lambda_motion,
            lambda_struct=self.lambda_struct,
            lambda_temp=self.lambda_temp,
            adaptive_weighting=self.adaptive_weighting,
            focal_gamma=self.focal_gamma,
        )
        self.validation_outputs = []
        self.best_val_iou = 0.0
        
        # Логирование
        self.log_images = train_config.get('log_images', True)
        self.log_interval = train_config.get('log_interval', 10)

        # Initialize grad scaler for mixed precision training
        self.scaler = torch.cuda.amp.GradScaler()
        
        # Convert model to float16
        self.model = self.model.half()
        
        # Ensure all buffers and parameters are float16
        for param in self.model.parameters():
            param.data = param.data.half()
        
        for buf in self.model.buffers():
            buf.data = buf.data.half()
        
        # Add a flag to track that we're using float16
        self.is_half = True
        
    def _get_model_dtype(self):
        """Get the dtype of model parameters"""
        return next(self.model.parameters()).dtype
        
    def forward(self, frames, target_bboxes):
        """Forward pass ensuring float16 precision"""
        # Convert inputs to float16
        with torch.cuda.amp.autocast(enabled=True, dtype=torch.float16):
            frames = frames.to(device=self.device, dtype=torch.float16)
            target_bboxes = target_bboxes.to(device=self.device, dtype=torch.float16)
            return self.model(frames, target_bboxes)
    
    def compute_iou(self, pred, target, threshold=0.5):
        """Compute IoU between predicted and target mask/bbox"""
        model_dtype = self._get_model_dtype()
        pred = pred.to(dtype=model_dtype)
        target = target.to(dtype=model_dtype)
        
        if self.input_type == 'mask':
            # Binarize predicted mask
            pred_bin = (pred > threshold).float()
            target_bin = (target > threshold).float()
            
            # Compute intersection and union
            intersection = (pred_bin * target_bin).sum((1, 2, 3))
            union = pred_bin.sum((1, 2, 3)) + target_bin.sum((1, 2, 3)) - intersection
            
            # IoU
            iou = intersection / (union + 1e-6)
            return iou.mean()
        else:
            # For bbox use the already implemented IoU function from loss
            giou_loss = self.loss_fn.compute_iou_loss(pred, target)
            return 1.0 - giou_loss  # Convert loss to metric
    
    def compute_bbox_metrics(self, pred, target):
        """Compute metrics for bbox: center, size, IoU"""
        model_dtype = self._get_model_dtype()
        pred = pred.to(dtype=model_dtype)
        target = target.to(dtype=model_dtype)
        
        # Extract center coordinates and dimensions
        pred_center_x, pred_center_y = pred[:, 0], pred[:, 1]
        pred_width, pred_height = pred[:, 2], pred[:, 3]
        
        target_center_x, target_center_y = target[:, 0], target[:, 1]
        target_width, target_height = target[:, 2], target[:, 3]
        
        # Compute errors
        center_error = torch.sqrt((pred_center_x - target_center_x)**2 + 
                                 (pred_center_y - target_center_y)**2)
        width_error = torch.abs(pred_width - target_width)
        height_error = torch.abs(pred_height - target_height)
        size_error = (width_error + height_error) / 2
        
        # IoU
        iou = self.compute_iou(pred, target)
        
        return {
            'center_error': center_error.mean(),
            'size_error': size_error.mean(),
            'iou': iou
        }
    
    def training_step(self, batch, batch_idx):
        # Enable autocast for mixed precision
        with torch.cuda.amp.autocast(enabled=True, dtype=torch.float16):
            # Move inputs to device (autocast handles precision)
            frames = batch['frames'].to(device=self.device)
            past = batch['past'].to(device=self.device)
            labels = batch['labels'].to(device=self.device)
            
            # Forward pass (will run in float16 where supported)
            predictions = self(frames, past)
            
            # Loss computation
            full_sequence = {
                'preds': predictions.unsqueeze(1) if self.input_type == 'mask' else predictions.unsqueeze(0),
                'targets': labels.unsqueeze(1) if self.input_type == 'mask' else labels.unsqueeze(0)
            }
            
            loss, loss_components = self.loss_fn(self.model, predictions, labels, full_sequence)
        
        # Log metrics outside autocast context (metrics use float32)
        iou = self.compute_iou(predictions, labels)
        self.log('train_iou', iou, on_step=True, on_epoch=True, prog_bar=True)
        
        if self.input_type == 'bbox':
            bbox_metrics = self.compute_bbox_metrics(predictions, labels)
            self.log('train_center_error', bbox_metrics['center_error'], on_step=True, on_epoch=True)
            self.log('train_size_error', bbox_metrics['size_error'], on_step=True, on_epoch=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        # Also use autocast for validation
        with torch.cuda.amp.autocast(enabled=True, dtype=torch.float16):
            # Move inputs to correct device and dtype
            frames = batch['frames'].to(device=self.device, dtype=torch.float16)
            past = batch['past'].to(device=self.device, dtype=torch.float16)
            labels = batch['labels'].to(device=self.device, dtype=torch.float16)
            
            # Forward pass
            predictions = self(frames, past)
            
            # Prepare sequence for temporal loss
            full_sequence = {
                'preds': predictions.unsqueeze(1) if self.input_type == 'mask' else predictions.unsqueeze(0),
                'targets': labels.unsqueeze(1) if self.input_type == 'mask' else labels.unsqueeze(0)
            }
            
            # Compute loss
            loss, loss_components = self.loss_fn(self.model, predictions, labels, full_sequence)
            
            # Log loss components
            for name, value in loss_components.items():
                self.log(f'val_{name}_loss', value, on_step=False, on_epoch=True)
            
            # Compute and log metrics
            iou = self.compute_iou(predictions, labels)
            self.log('val_iou', iou, on_step=False, on_epoch=True, prog_bar=True)
            self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
            
            if self.input_type == 'bbox':
                bbox_metrics = self.compute_bbox_metrics(predictions, labels)
                self.log('val_center_error', bbox_metrics['center_error'], on_step=False, on_epoch=True)
                self.log('val_size_error', bbox_metrics['size_error'], on_step=False, on_epoch=True)
            
            # Save for visualization
            if batch_idx % self.log_interval == 0 and self.log_images:
                self.validation_outputs.append({
                    'frames': frames,
                    'past': past,
                    'labels': labels,
                    'predictions': predictions,
                    'batch_idx': batch_idx
                })
            
            return {'val_loss': loss, 'val_iou': iou}
    
    def on_validation_epoch_end(self):
        """Действия в конце эпохи валидации"""
        # Обработка сохраненных выходов для визуализации
        if self.log_images and self.validation_outputs and self.global_rank == 0:
            self._log_images()
        
        self.validation_outputs = []
    
    def _log_images(self):
        """Логирует изображения в TensorBoard/W&B"""
        # Берем первый батч из сохраненных выходов
        outputs = self.validation_outputs[0]
        frames = outputs['frames']
        past = outputs['past']
        labels = outputs['labels']
        predictions = outputs['predictions']
        
        # Выбираем до 4 примеров из батча
        n_examples = min(4, frames.shape[0])
        
        if self.input_type == 'mask':
            # Создаем фигуру для каждого примера
            fig, axes = plt.subplots(n_examples, 3, figsize=(12, 3*n_examples))
            if n_examples == 1:
                axes = axes.reshape(1, -1)
            
            for i in range(n_examples):
                # Последний кадр
                last_frame = frames[i, -1].cpu().permute(1, 2, 0).numpy()
                if last_frame.max() > 1.0:
                    last_frame = last_frame / 255.0
                axes[i, 0].imshow(last_frame)
                axes[i, 0].set_title("Последний кадр")
                axes[i, 0].axis('off')
                
                # Целевая маска
                target_mask = labels[i, 0].cpu().numpy()
                axes[i, 1].imshow(target_mask, cmap='jet')
                axes[i, 1].set_title("Целевая маска")
                axes[i, 1].axis('off')
                
                # Предсказанная маска
                pred_mask = predictions[i, 0].cpu().detach().numpy()
                axes[i, 2].imshow(pred_mask, cmap='jet')
                axes[i, 2].set_title("Предсказанная маска")
                axes[i, 2].axis('off')
            
            # Сохраняем фигуру
            plt.tight_layout()
            
            # Логируем в TensorBoard/W&B если доступно
            if isinstance(self.logger, TensorBoardLogger):
                self.logger.experiment.add_figure('val_masks', fig, global_step=self.global_step)
            elif isinstance(self.logger, WandbLogger):
                import wandb
                self.logger.experiment.log({"val_masks": wandb.Image(fig)})
            
            plt.close(fig)
        else:
            # Для bbox, рисуем bbox на последнем кадре
            fig, axes = plt.subplots(n_examples, 1, figsize=(6, 4*n_examples))
            if n_examples == 1:
                axes = [axes]
            
            for i in range(n_examples):
                # Последний кадр
                last_frame = frames[i, -1].cpu().permute(1, 2, 0).numpy()
                if last_frame.max() > 1.0:
                    last_frame = last_frame / 255.0
                axes[i].imshow(last_frame)
                
                # Получаем размеры кадра
                height, width = last_frame.shape[:2]
                
                # Целевой bbox - нормализованные координаты [x, y, w, h]
                target_bbox = labels[i].cpu().numpy()
                tx, ty, tw, th = target_bbox
                
                # Преобразуем центр и размер в координаты углов
                tx1, ty1 = int((tx - tw/2) * width), int((ty - th/2) * height)
                tx2, ty2 = int((tx + tw/2) * width), int((ty + th/2) * height)
                
                # Рисуем целевой bbox
                from matplotlib.patches import Rectangle
                target_rect = Rectangle((tx1, ty1), tx2-tx1, ty2-ty1, 
                                      linewidth=2, edgecolor='r', facecolor='none')
                axes[i].add_patch(target_rect)
                
                # Предсказанный bbox
                pred_bbox = predictions[i].cpu().detach().numpy()
                px, py, pw, ph = pred_bbox
                
                # Преобразуем центр и размер в координаты углов
                px1, py1 = int((px - pw/2) * width), int((py - ph/2) * height)
                px2, py2 = int((px + pw/2) * width), int((py + ph/2) * height)
                
                # Рисуем предсказанный bbox
                pred_rect = Rectangle((px1, py1), px2-px1, py2-py1, 
                                     linewidth=2, edgecolor='g', facecolor='none')
                axes[i].add_patch(pred_rect)
                
                axes[i].set_title(f"Sample {i}: Target (red) vs Prediction (green)")
                axes[i].axis('off')
            
            # Сохраняем фигуру
            plt.tight_layout()
            
            # Логируем в TensorBoard/W&B если доступно
            if isinstance(self.logger, TensorBoardLogger):
                self.logger.experiment.add_figure('val_bboxes', fig, global_step=self.global_step)
            elif isinstance(self.logger, WandbLogger):
                import wandb
                self.logger.experiment.log({"val_bboxes": wandb.Image(fig)})
            
            plt.close(fig)
    
    def configure_optimizers(self):
        # Return optimizer only (scaler handles scaling)
        optimizer = torch.optim.AdamW(
            self.parameters(), 
            lr=self.learning_rate, 
            weight_decay=self.weight_decay
        )
        return optimizer


def lasot_collate_fn(batch):
    """
    Custom collate function for LaSotDataset that handles variable-length sequences.
    
    Args:
        batch: List of dictionaries, each containing 'frames', 'past', and 'labels'
        
    Returns:
        Dictionary with batched and padded tensors
    """
    # Find the maximum sequence length in this batch
    max_seq_len = max([item['frames'].shape[0] for item in batch])
    
    batch_size = len(batch)
    # Get dimensions from the first item
    _, C, H, W = batch[0]['frames'].shape
    
    # Initialize tensors for the batch
    frames_batch = torch.zeros(batch_size, max_seq_len, C, H, W)
    past_batch = torch.zeros(batch_size, max_seq_len, 4)
    labels_batch = torch.zeros(batch_size, 4)
    
    # Fill the batch tensors
    for i, item in enumerate(batch):
        frames = item['frames']
        past = item['past']
        seq_len = frames.shape[0]
        
        # Copy data
        frames_batch[i, :seq_len] = frames
        past_batch[i, :seq_len] = past
        labels_batch[i] = item['labels']
    
    return {
        'frames': frames_batch,       # Shape: [B, max_seq_len, C, H, W]
        'past': past_batch,           # Shape: [B, max_seq_len, 4]
        'labels': labels_batch,       # Shape: [B, 4]
    }