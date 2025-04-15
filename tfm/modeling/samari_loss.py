import torch
import torch.nn as nn
import torch.nn.functional as F


class KalmanLoss(nn.Module):
    """
    Улучшенная функция потерь для обучения LearnableKalmanTracker
    с акцентом на предсказание и робастность
    """

    def __init__(
        self, prediction_weight=1.0, smoothness_weight=0.1, forecast_weight=2.0
    ):
        super().__init__()
        self.prediction_weight = prediction_weight
        self.smoothness_weight = smoothness_weight
        self.forecast_weight = forecast_weight

    def forward(self, pred_boxes, target_boxes, prediction_only_boxes=None, valid_mask=None):
        """
        Вычисляет потери с учетом маски действительных элементов
        
        Args:
            pred_boxes: предсказанные боксы [batch_size, seq_len, 4]
            target_boxes: целевые боксы [batch_size, seq_len, 4]
            prediction_only_boxes: боксы, полученные только предсказанием [batch_size, seq_len-1, 4]
            valid_mask: маска действительных (не паддинговых) элементов [batch_size, seq_len]
        """
        if valid_mask is None:
            # Если маска не передана, считаем все элементы действительными
            valid_mask = torch.ones(pred_boxes.shape[0], pred_boxes.shape[1], 
                                    dtype=torch.bool, device=pred_boxes.device)
        
        # Считаем L1 потерю только на действительных элементах
        # Создаем маску для боксов (расширяем valid_mask)
        box_mask = valid_mask.unsqueeze(-1).expand_as(pred_boxes)
        
        # Маскируем предсказания и цели
        pred_masked = pred_boxes * box_mask
        target_masked = target_boxes * box_mask
        
        # Считаем потерю с учетом количества действительных элементов
        prediction_loss = F.smooth_l1_loss(pred_masked, target_masked, reduction='sum')
        num_valid = valid_mask.sum()
        if num_valid > 0:
            prediction_loss = prediction_loss / num_valid
        
        # Сглаженная потеря для плавности изменений
        if pred_boxes.shape[1] > 1:
            # Маска для разности (нужно исключить разницу между паддинговым и последним действительным кадром)
            valid_diff_mask = valid_mask[:, 1:] & valid_mask[:, :-1]
            box_diff_mask = valid_diff_mask.unsqueeze(-1).expand(-1, -1, 4)
            
            # Маскированные разности
            pred_diff = (pred_boxes[:, 1:] - pred_boxes[:, :-1]) * box_diff_mask
            target_diff = (target_boxes[:, 1:] - target_boxes[:, :-1]) * box_diff_mask
            
            # Считаем потерю плавности
            smoothness_loss = F.smooth_l1_loss(pred_diff, target_diff, reduction='sum')
            num_valid_diffs = valid_diff_mask.sum()
            if num_valid_diffs > 0:
                smoothness_loss = smoothness_loss / num_valid_diffs
            else:
                smoothness_loss = torch.tensor(0.0, device=pred_boxes.device)
        else:
            smoothness_loss = torch.tensor(0.0, device=pred_boxes.device)
        
        # Потеря предсказания
        if prediction_only_boxes is not None:
            # Маска для предсказаний (со 2-го кадра)
            forecast_mask = valid_mask[:, 1:].unsqueeze(-1).expand_as(prediction_only_boxes)
            forecast_masked = prediction_only_boxes * forecast_mask
            target_forecast_masked = target_boxes[:, 1:] * forecast_mask
            
            forecast_loss = F.smooth_l1_loss(forecast_masked, target_forecast_masked, reduction='sum')
            num_valid_forecasts = valid_mask[:, 1:].sum()
            if num_valid_forecasts > 0:
                forecast_loss = forecast_loss / num_valid_forecasts
            else:
                forecast_loss = torch.tensor(0.0, device=pred_boxes.device)
        else:
            forecast_loss = torch.tensor(0.0, device=pred_boxes.device)
        
        # Комбинируем потери
        total_loss = (self.prediction_weight * prediction_loss + 
                      self.smoothness_weight * smoothness_loss +
                      self.forecast_weight * forecast_loss)
                  
        return total_loss
