import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class STMTKFLoss(nn.Module):
    """
    Продвинутый комплексный loss для модели Space-Time Mamba Temporal Kalman Filter (STMTKF).
    
    Включает компоненты:
    1. Основной loss (маски/bbox)
    2. Временная согласованность
    3. Калмановская регуляризация
    4. Структурный loss
    5. Адаптивное взвешивание
    """
    
    def __init__(
        self,
        input_type="mask",
        lambda_temp=0.5,
        lambda_struct=0.3,
        lambda_kf=0.2,
        lambda_motion=0.4,
        adaptive_weighting=True,
        focal_gamma=2.0,
        device=None
    ):
        super().__init__()
        
        self.input_type = input_type
        self.lambda_temp = lambda_temp
        self.lambda_struct = lambda_struct
        self.lambda_kf = lambda_kf
        self.lambda_motion = lambda_motion
        self.adaptive_weighting = adaptive_weighting
        self.focal_gamma = focal_gamma
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # История обучения для адаптивного взвешивания
        self.loss_history = {
            "main": [],
            "temporal": [],
            "structural": [],
            "kf": [],
            "motion": []
        }
        
        # Граничные значения для нормализации loss компонентов
        self.eps = 1e-6
        
    def compute_dice_loss(self, pred, target):
        """
        Вычисляет Dice loss между предсказанными и целевыми масками.
        Dice = (2*|X∩Y|)/(|X|+|Y|)
        """
        # Сглаживание для численной стабильности
        smooth = 1.0
        
        # Уплощаем маски
        pred_flat = pred.reshape(-1)
        target_flat = target.reshape(-1)
        
        # Вычисляем пересечение и объединение
        intersection = (pred_flat * target_flat).sum()
        denominator = pred_flat.sum() + target_flat.sum()
        
        # Dice coefficient и loss
        dice = (2.0 * intersection + smooth) / (denominator + smooth)
        return 1.0 - dice
    
    def compute_iou_loss(self, pred_box, target_box):
        """
        Вычисляет Generalized IoU loss для bounding boxes.
        
        pred_box и target_box имеют формат [x, y, w, h], где:
        - (x, y) - центр бокса
        - (w, h) - ширина и высота
        """
        # Конвертация из x,y,w,h в x1,y1,x2,y2
        pred_x1 = pred_box[:, 0] - pred_box[:, 2] / 2
        pred_y1 = pred_box[:, 1] - pred_box[:, 3] / 2
        pred_x2 = pred_box[:, 0] + pred_box[:, 2] / 2
        pred_y2 = pred_box[:, 1] + pred_box[:, 3] / 2
        
        target_x1 = target_box[:, 0] - target_box[:, 2] / 2
        target_y1 = target_box[:, 1] - target_box[:, 3] / 2
        target_x2 = target_box[:, 0] + target_box[:, 2] / 2
        target_y2 = target_box[:, 1] + target_box[:, 3] / 2
        
        # Площади боксов
        pred_area = pred_box[:, 2] * pred_box[:, 3]
        target_area = target_box[:, 2] * target_box[:, 3]
        
        # Координаты пересечения
        inter_x1 = torch.max(pred_x1, target_x1)
        inter_y1 = torch.max(pred_y1, target_y1)
        inter_x2 = torch.min(pred_x2, target_x2)
        inter_y2 = torch.min(pred_y2, target_y2)
        
        # Площадь пересечения
        inter_w = torch.clamp(inter_x2 - inter_x1, min=0)
        inter_h = torch.clamp(inter_y2 - inter_y1, min=0)
        intersection = inter_w * inter_h
        
        # IoU
        union = pred_area + target_area - intersection
        iou = intersection / (union + self.eps)
        
        # Координаты наименьшего описывающего бокса
        enclosing_x1 = torch.min(pred_x1, target_x1)
        enclosing_y1 = torch.min(pred_y1, target_y1)
        enclosing_x2 = torch.max(pred_x2, target_x2)
        enclosing_y2 = torch.max(pred_y2, target_y2)
        
        # Площадь описывающего бокса
        enclosing_w = enclosing_x2 - enclosing_x1
        enclosing_h = enclosing_y2 - enclosing_y1
        enclosing_area = enclosing_w * enclosing_h
        
        # GIoU = IoU - (area(C) - area(A∪B)) / area(C)
        giou = iou - (enclosing_area - union) / (enclosing_area + self.eps)
        
        # GIoU loss
        giou_loss = 1.0 - giou
        return giou_loss.mean()
    
    def compute_focal_loss(self, pred, target):
        """
        Focal Loss для масок - эффективнее обрабатывает классовый дисбаланс.
        """
        bce_loss = F.binary_cross_entropy(pred, target, reduction='none')
        pt = torch.exp(-bce_loss)  # pt = p для target=1, pt = 1-p для target=0
        focal_weight = (1 - pt) ** self.focal_gamma
        
        return (focal_weight * bce_loss).mean()
    
    def compute_temporal_consistency_loss(self, preds, targets):
        """
        Loss для временной согласованности - поощряет плавные изменения
        между последовательными кадрами.
        """
        if self.input_type == "mask":
            # Для масок: сравниваем разницу между последовательными предсказаниями 
            # и разницу между последовательными целями
            B, T, H, W = targets.shape
            
            # Вычисляем разницы
            pred_diffs = preds[:, 1:] - preds[:, :-1]
            target_diffs = targets[:, 1:] - targets[:, :-1]
            
            # Сравниваем паттерны изменений
            temp_loss = F.mse_loss(pred_diffs, target_diffs)
            
        else:  # bbox
            # Для bbox: поощряем плавные изменения в предсказаниях
            B, T, _ = targets.shape
            
            # Вычисляем разницы и ускорения
            pred_velocities = preds[:, 1:] - preds[:, :-1]
            target_velocities = targets[:, 1:] - targets[:, :-1]
            
            # Сравниваем скорости
            velocity_loss = F.mse_loss(pred_velocities, target_velocities)
            
            # Ускорения (если есть хотя бы 3 кадра)
            if T >= 3:
                pred_accel = pred_velocities[:, 1:] - pred_velocities[:, :-1]
                target_accel = target_velocities[:, 1:] - target_velocities[:, :-1]
                accel_loss = F.mse_loss(pred_accel, target_accel)
                temp_loss = velocity_loss + 0.5 * accel_loss
            else:
                temp_loss = velocity_loss
                
        return temp_loss
    
    def compute_structural_loss(self, pred, target):
        """
        Структурный loss - сохраняет пространственную целостность объекта.
        Использует градиентные разницы для сохранения границ.
        """
        if self.input_type == "mask":
            # Sobel-фильтры для выделения градиентов
            sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], 
                                  dtype=torch.float32, device=self.device).unsqueeze(0).unsqueeze(0)
            sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], 
                                  dtype=torch.float32, device=self.device).unsqueeze(0).unsqueeze(0)
            
            # Если pred имеет формат [B, 1, H, W]
            if pred.dim() == 4:
                B, C, H, W = pred.shape
                
                # Применяем фильтры
                pred_grad_x = F.conv2d(pred, sobel_x, padding=1)
                pred_grad_y = F.conv2d(pred, sobel_y, padding=1)
                target_grad_x = F.conv2d(target, sobel_x, padding=1)
                target_grad_y = F.conv2d(target, sobel_y, padding=1)
                
                # Структурная разница
                grad_diff_x = F.mse_loss(pred_grad_x, target_grad_x)
                grad_diff_y = F.mse_loss(pred_grad_y, target_grad_y)
                
                struct_loss = grad_diff_x + grad_diff_y
            else:
                struct_loss = torch.tensor(0.0, device=self.device)
        else:  # bbox
            # Для bbox структурный loss не имеет смысла
            struct_loss = torch.tensor(0.0, device=self.device)
            
        return struct_loss
    
    def compute_kalman_regularization(self, model, pred, target):
        """
        Регуляризует параметры фильтра Калмана для улучшения предсказаний.
        """
        # Получаем матрицу перехода F из модели
        F = getattr(model, 'F', None)
        if F is not None:
            # Поощряем стабильность матрицы F (eigenvalues < 1)
            eye = torch.eye(F.shape[-1], device=F.device)
            f_regularizer = torch.norm(F - eye, p='fro')
            
            # Если eigenvalues > 1, повышаем штраф
            eigenvalues = torch.linalg.eigvals(F)
            unstable = torch.relu(torch.abs(eigenvalues) - 1.0)
            stability_penalty = unstable.sum()
            
            kf_loss = f_regularizer + 10.0 * stability_penalty
        else:
            kf_loss = torch.tensor(0.0, device=self.device)
            
        return kf_loss
    
    def compute_motion_loss(self, pred_sequence, target_sequence):
        """
        Специализированный loss для анализа движения объекта.
        Сравнивает траектории и скорости.
        """
        if self.input_type == "bbox":
            # Центральные точки
            pred_centers = pred_sequence[:, :, :2]  # [B, T, 2] (x, y)
            target_centers = target_sequence[:, :, :2]  # [B, T, 2] (x, y)
            
            # Траектория
            trajectory_loss = F.mse_loss(pred_centers, target_centers)
            
            # Форма
            pred_shapes = pred_sequence[:, :, 2:]  # [B, T, 2] (w, h)
            target_shapes = target_sequence[:, :, 2:]  # [B, T, 2] (w, h)
            shape_loss = F.mse_loss(pred_shapes, target_shapes)
            
            # Скорость и направление
            if pred_sequence.shape[1] > 1:
                pred_velocity = pred_centers[:, 1:] - pred_centers[:, :-1]
                target_velocity = target_centers[:, 1:] - target_centers[:, :-1]
                
                velocity_magnitude_pred = torch.norm(pred_velocity, dim=2)
                velocity_magnitude_target = torch.norm(target_velocity, dim=2)
                
                # Направление - нормализованные векторы
                pred_direction = pred_velocity / (velocity_magnitude_pred.unsqueeze(-1) + self.eps)
                target_direction = target_velocity / (velocity_magnitude_target.unsqueeze(-1) + self.eps)
                
                # Сходство направлений (косинусное сходство)
                direction_similarity = torch.sum(pred_direction * target_direction, dim=2)
                direction_loss = 1.0 - direction_similarity.mean()
                
                # Разница в скорости
                speed_loss = F.mse_loss(velocity_magnitude_pred, velocity_magnitude_target)
                
                motion_loss = trajectory_loss + 0.5 * shape_loss + 0.3 * direction_loss + 0.2 * speed_loss
            else:
                motion_loss = trajectory_loss + shape_loss
        else:  # mask
            # Для масок - упрощенный анализ движения через оптический поток
            # В реальной реализации можно использовать предварительно рассчитанный оптический поток
            motion_loss = torch.tensor(0.0, device=self.device)
            
        return motion_loss
    
    def adaptive_weight_update(self, losses):
        """
        Динамически обновляет веса компонентов loss на основе их истории.
        Использует принцип "уделить больше внимания сложным компонентам".
        """
        # Сохраняем историю
        for k, v in losses.items():
            if k != 'total':
                self.loss_history[k].append(v.item())
                if len(self.loss_history[k]) > 50:  # Окно истории
                    self.loss_history[k].pop(0)
        
        # Если истории недостаточно, используем стандартные веса
        if len(self.loss_history['main']) < 10:
            return {
                'main': 1.0,
                'temporal': self.lambda_temp,
                'structural': self.lambda_struct,
                'kf': self.lambda_kf,
                'motion': self.lambda_motion
            }
        
        # Вычисляем средние значения и стандартные отклонения
        means = {k: np.mean(v) if v else 0 for k, v in self.loss_history.items()}
        stds = {k: np.std(v) if v else 1 for k, v in self.loss_history.items()}
        
        # Нормализуем значения для получения весов
        total = sum(means.values()) + self.eps
        base_weights = {k: v / total for k, v in means.items()}
        
        # Корректируем веса с учетом вариации (более вариативные компоненты получают больший вес)
        variation = {k: s / (means[k] + self.eps) for k, s in stds.items()}
        total_var = sum(variation.values()) + self.eps
        var_weights = {k: v / total_var for k, v in variation.items()}
        
        # Комбинируем базовые веса и вариационные коэффициенты
        adaptive_weights = {
            'main': 1.0,  # Основной loss всегда имеет вес 1.0
            'temporal': self.lambda_temp * (1.0 + var_weights['temporal']),
            'structural': self.lambda_struct * (1.0 + var_weights['structural']),
            'kf': self.lambda_kf * (1.0 + var_weights['kf']),
            'motion': self.lambda_motion * (1.0 + var_weights['motion'])
        }
        
        return adaptive_weights
        
    def forward(self, model, predictions, targets, full_sequence=None):
        """
        Вычисляет комбинированный loss.
        
        Args:
            model: модель STMTKF для доступа к внутренним параметрам
            predictions: предсказания модели
            targets: целевые значения
            full_sequence: полная последовательность для временного loss (опционально)
        
        Returns:
            total_loss: финальный взвешенный loss
        """
        losses = {}
        
        # Основной loss в зависимости от типа задачи
        if self.input_type == "mask":
            # Для масок комбинируем Dice и Focal losses
            dice_loss = self.compute_dice_loss(predictions, targets)
            focal_loss = self.compute_focal_loss(predictions, targets)
            losses['main'] = 0.5 * dice_loss + 0.5 * focal_loss
        else:  # bbox
            # Для bbox используем GIoU loss
            losses['main'] = self.compute_iou_loss(predictions, targets)
        
        # Временная согласованность (если доступна полная последовательность)
        if full_sequence is not None:
            losses['temporal'] = self.compute_temporal_consistency_loss(
                full_sequence['preds'], full_sequence['targets']
            )
        else:
            losses['temporal'] = torch.tensor(0.0, device=self.device)
        
        # Структурный loss
        losses['structural'] = self.compute_structural_loss(predictions, targets)
        
        # Калмановская регуляризация
        losses['kf'] = self.compute_kalman_regularization(model, predictions, targets)
        
        # Анализ движения
        if full_sequence is not None:
            losses['motion'] = self.compute_motion_loss(
                full_sequence['preds'], full_sequence['targets']
            )
        else:
            losses['motion'] = torch.tensor(0.0, device=self.device)
        
        # Применяем веса к компонентам loss
        if self.adaptive_weighting:
            weights = self.adaptive_weight_update(losses)
        else:
            weights = {
                'main': 1.0,
                'temporal': self.lambda_temp,
                'structural': self.lambda_struct,
                'kf': self.lambda_kf,
                'motion': self.lambda_motion
            }
        
        # Вычисляем итоговый loss
        total_loss = sum(w * losses[k] for k, w in weights.items())
        losses['total'] = total_loss
        
        # Отладочная информация
        with torch.no_grad():
            self.last_losses = {k: v.item() for k, v in losses.items()}
            self.last_weights = weights
        
        return total_loss, losses


# Пример использования
if __name__ == "__main__":
    # Создаем синтетические данные
    B, T, H, W = 2, 5, 64, 64  # Batch, Time, Height, Width
    
    # Маски
    pred_masks = torch.rand(B, 1, H//2, W//2, requires_grad=True)
    target_masks = torch.rand(B, 1, H//2, W//2)
    
    # Последовательности масок
    pred_mask_seq = torch.rand(B, T, 1, H//2, W//2, requires_grad=True)
    target_mask_seq = torch.rand(B, T, 1, H//2, W//2)
    
    # Bbox
    pred_boxes = torch.rand(B, 4, requires_grad=True)
    target_boxes = torch.rand(B, 4)
    
    # Последовательности bbox
    pred_box_seq = torch.rand(B, T, 4, requires_grad=True)
    target_box_seq = torch.rand(B, T, 4)
    
    # Move all tensor to device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pred_masks = pred_masks.to(device)
    target_masks = target_masks.to(device)
    pred_box_seq = pred_box_seq.to(device)
    target_box_seq = target_box_seq.to(device)
    pred_boxes = pred_boxes.to(device)
    target_boxes = target_boxes.to(device)

    # Тестируем loss для масок
    mask_loss = STMTKFLoss(input_type="mask")
    mask_sequences = {
        'preds': pred_mask_seq.squeeze(2),
        'targets': target_mask_seq.squeeze(2)
    }
    total_mask_loss, mask_components = mask_loss(None, pred_masks, target_masks, mask_sequences)
    print("Mask Loss Components:", mask_components)
    
    # Тестируем loss для bbox
    bbox_loss = STMTKFLoss(input_type="bbox")
    bbox_sequences = {
        'preds': pred_box_seq,
        'targets': target_box_seq
    }
    total_bbox_loss, bbox_components = bbox_loss(None, pred_boxes, target_boxes, bbox_sequences)
    print("Bbox Loss Components:", bbox_components)