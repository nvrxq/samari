import torch
from tfm.modeling.samari import LearnableKalmanTracker
from tfm.dataset.samset import SavSamariDataset
from tfm.modeling.samari_loss import KalmanLoss
import os
import torch.nn.functional as F
import wandb


def train_kalman_tracker(
    model,
    train_loader,
    val_loader=None,
    num_epochs=20,
    lr=0.001,
    weight_decay=1e-5,
    patience=5,
    device="cuda",
    save_path="models/kalman_tracker.pth",
    load_checkpoint=None,
    use_wandb=True,
    project_name="kalman-tracker",
    experiment_name=None,
):
    """
    Обучает LearnableKalmanTracker на данных с шумом и окклюзиями

    Args:
        model: модель LearnableKalmanTracker
        train_loader: загрузчик тренировочных данных
        val_loader: загрузчик валидационных данных (может быть None)
        num_epochs: количество эпох обучения
        lr: скорость обучения
        weight_decay: L2 регуляризация
        patience: количество эпох без улучшения для early stopping
        device: устройство для обучения ('cuda' или 'cpu')
        save_path: путь для сохранения лучшей модели
        load_checkpoint: путь к чекпоинту для продолжения обучения (или None)
        use_wandb: использовать ли WandB для логирования
        project_name: имя проекта WandB
        experiment_name: имя эксперимента WandB

    Returns:
        trained_model: обученная модель
    """
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3
    )
    criterion = KalmanLoss(
        prediction_weight=1.0, smoothness_weight=0.1, forecast_weight=2.0
    )

    # Инициализация WandB
    if use_wandb:
        if experiment_name is None:
            experiment_name = f"kalman_tracker_{lr}_{weight_decay}"
        
        wandb.init(project=project_name, name=experiment_name)
        wandb.config.update({
            "lr": lr,
            "weight_decay": weight_decay,
            "epochs": num_epochs,
            "patience": patience,
            "prediction_weight": criterion.prediction_weight,
            "smoothness_weight": criterion.smoothness_weight,
            "forecast_weight": criterion.forecast_weight,
        })
        # Логируем модель
        wandb.watch(model)

    # Загружаем чекпоинт, если указан
    start_epoch = 0
    best_val_loss = float("inf")

    if load_checkpoint is not None and os.path.exists(load_checkpoint):
        checkpoint = torch.load(load_checkpoint)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        best_val_loss = checkpoint["best_val_loss"]
        print(
            f"Загружен чекпоинт: эпоха {start_epoch-1}, лучшая валидационная потеря: {best_val_loss:.4f}"
        )

    # Создаем директорию для сохранения моделей, если её нет
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Для early stopping
    no_improve_epochs = 0

    # Основной цикл обучения
    for epoch in range(start_epoch, start_epoch + num_epochs):
        # ТРЕНИРОВКА
        model.train()
        train_loss = 0.0
        train_prediction_loss = 0.0
        train_forecast_loss = 0.0

        for batch_idx, batch in enumerate(train_loader):
            # Получаем тензоры с устройства
            noisy_boxes = batch['boxes'].to(device)
            target_boxes = batch['original_boxes'].to(device)
            valid_mask = batch['valid_mask'].to(device)
            seq_lengths = batch['seq_lengths'].to(device)
            
            # Получаем маску окклюзий (если есть)
            if 'occlusion_mask' in batch:
                occlusion_mask = batch['occlusion_mask'].to(device)
            else:
                occlusion_mask = torch.ones((noisy_boxes.shape[0], noisy_boxes.shape[1]), device=device)
            
            batch_size, seq_len, _ = noisy_boxes.shape
            all_predictions = []
            all_forecasts = []

            for b in range(batch_size):
                # Сбрасываем состояние модели
                model.reset()
                sequence_predictions = []
                sequence_forecasts = []

                # Инициализируем с первым кадром (всегда используем)
                model(noisy_boxes[b, 0])
                sequence_predictions.append(
                    noisy_boxes[b, 0]
                )  # Первый кадр используем как есть

                # Для остальных кадров
                for t in range(1, seq_len):
                    # Делаем предсказание без измерения
                    forecast_box = model.predict()
                    sequence_forecasts.append(forecast_box)

                    if occlusion_mask[b, t] > 0:
                        # Если измерение доступно, используем его
                        pred_box = model(noisy_boxes[b, t])
                    else:
                        # Если окклюзия, используем только предсказание
                        pred_box = forecast_box

                    sequence_predictions.append(pred_box)

                all_predictions.append(torch.stack(sequence_predictions))
                if sequence_forecasts:
                    all_forecasts.append(torch.stack(sequence_forecasts))

            # Собираем предсказания в батч
            pred_boxes = torch.stack(all_predictions)

            # Оптимизация
            optimizer.zero_grad()

            if all_forecasts:
                forecast_boxes = torch.stack(all_forecasts)
                
                # Применяем маску действительных элементов
                pred_valid = pred_boxes * valid_mask.unsqueeze(-1)
                target_valid = target_boxes * valid_mask.unsqueeze(-1)
                forecast_valid = forecast_boxes * valid_mask[:, 1:].unsqueeze(-1)
                
                # Вычисляем потери только на действительных элементах
                loss = criterion(pred_valid, target_valid, forecast_valid, valid_mask)

                # Для мониторинга компонентов потерь
                with torch.no_grad():
                    pred_loss = F.l1_loss(pred_valid, target_valid)
                    forecast_loss = F.l1_loss(forecast_valid, target_valid[:, 1:])
                    train_prediction_loss += pred_loss.item()
                    train_forecast_loss += forecast_loss.item()
            else:
                # Только для предсказаний и целевых боксов
                pred_valid = pred_boxes * valid_mask.unsqueeze(-1)
                target_valid = target_boxes * valid_mask.unsqueeze(-1)
                loss = criterion(pred_valid, target_valid, valid_mask=valid_mask)
                with torch.no_grad():
                    train_prediction_loss += F.l1_loss(pred_valid, target_valid).item()

            loss.backward()

            # Градиентный клиппинг для стабильности
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            train_loss += loss.item()

            # Выводим прогресс каждые 10 батчей
            if (batch_idx + 1) % 10 == 0:
                print(
                    f"Epoch {epoch+1}/{start_epoch+num_epochs}, Batch {batch_idx+1}/{len(train_loader)}, Loss: {loss.item():.4f}"
                )
                # Логируем промежуточные результаты в WandB
                if use_wandb:
                    wandb.log({
                        "batch": batch_idx + epoch * len(train_loader),
                        "train_batch_loss": loss.item(),
                    })

        # Считаем средние потери за эпоху
        train_loss /= len(train_loader)
        train_prediction_loss /= len(train_loader)
        train_forecast_loss /= len(train_loader)

        print(
            f"Эпоха {epoch+1}/{start_epoch+num_epochs}, Потеря обучения: {train_loss:.4f}, "
            f"Pred: {train_prediction_loss:.4f}, Forecast: {train_forecast_loss:.4f}"
        )

        # Логируем метрики тренировки в WandB
        if use_wandb:
            wandb.log({
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "train_prediction_loss": train_prediction_loss,
                "train_forecast_loss": train_forecast_loss,
                "learning_rate": optimizer.param_groups[0]['lr'],
            })

        # ВАЛИДАЦИЯ
        val_loss = float("inf")

        if val_loader is not None:
            model.eval()
            val_loss = 0.0
            val_prediction_loss = 0.0
            val_forecast_loss = 0.0

            with torch.no_grad():
                for batch in val_loader:
                    # Повторяем ту же логику, что и для обучения
                    noisy_boxes = batch['boxes'].to(device)
                    target_boxes = batch['original_boxes'].to(device)
                    valid_mask = batch['valid_mask'].to(device)
                    seq_lengths = batch['seq_lengths'].to(device)

                    if 'occlusion_mask' in batch:
                        occlusion_mask = batch['occlusion_mask'].to(device)
                    else:
                        occlusion_mask = torch.ones((noisy_boxes.shape[0], noisy_boxes.shape[1]), device=device)

                    batch_size, seq_len, _ = noisy_boxes.shape
                    all_predictions = []
                    all_forecasts = []

                    for b in range(batch_size):
                        model.reset()
                        sequence_predictions = []
                        sequence_forecasts = []

                        model(noisy_boxes[b, 0])
                        sequence_predictions.append(noisy_boxes[b, 0])

                        for t in range(1, seq_len):
                            forecast_box = model.predict()
                            sequence_forecasts.append(forecast_box)

                            if occlusion_mask[b, t] > 0:
                                pred_box = model(noisy_boxes[b, t])
                            else:
                                pred_box = forecast_box

                            sequence_predictions.append(pred_box)

                        all_predictions.append(torch.stack(sequence_predictions))
                        if sequence_forecasts:
                            all_forecasts.append(torch.stack(sequence_forecasts))

                    pred_boxes = torch.stack(all_predictions)

                    if all_forecasts:
                        forecast_boxes = torch.stack(all_forecasts)
                        
                        # Применяем маску действительных элементов
                        pred_valid = pred_boxes * valid_mask.unsqueeze(-1)
                        target_valid = target_boxes * valid_mask.unsqueeze(-1)
                        forecast_valid = forecast_boxes * valid_mask[:, 1:].unsqueeze(-1)
                        
                        # Вычисляем потери только на действительных элементах
                        batch_loss = criterion(pred_valid, target_valid, forecast_valid, valid_mask)
                        pred_loss = F.l1_loss(pred_valid, target_valid)
                        forecast_loss = F.l1_loss(forecast_valid, target_valid[:, 1:])
                        val_prediction_loss += pred_loss.item()
                        val_forecast_loss += forecast_loss.item()
                    else:
                        # Только для предсказаний и целевых боксов
                        pred_valid = pred_boxes * valid_mask.unsqueeze(-1)
                        target_valid = target_boxes * valid_mask.unsqueeze(-1)
                        batch_loss = criterion(pred_valid, target_valid, valid_mask=valid_mask)
                        val_prediction_loss += F.l1_loss(
                            pred_valid, target_valid
                        ).item()

                    val_loss += batch_loss.item()

            val_loss /= len(val_loader)
            val_prediction_loss /= len(val_loader)
            val_forecast_loss /= len(val_loader)

            print(
                f"Валидационная потеря: {val_loss:.4f}, "
                f"Pred: {val_prediction_loss:.4f}, Forecast: {val_forecast_loss:.4f}"
            )

            # Логируем метрики валидации в WandB
            if use_wandb:
                wandb.log({
                    "epoch": epoch + 1,
                    "val_loss": val_loss,
                    "val_prediction_loss": val_prediction_loss,
                    "val_forecast_loss": val_forecast_loss,
                })

            # Обновляем learning rate scheduler
            scheduler.step(val_loss)

            # Сохраняем лучшую модель
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                print(
                    f"Сохраняем лучшую модель с валидационной потерей: {best_val_loss:.4f}"
                )

                checkpoint = {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_val_loss": best_val_loss,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                }

                torch.save(checkpoint, save_path)
                
                # Логируем модель в WandB
                if use_wandb:
                    wandb.save(save_path)
                    
                no_improve_epochs = 0
            else:
                no_improve_epochs += 1
                print(f"Нет улучшения {no_improve_epochs} эпох")

                if no_improve_epochs >= patience:
                    print(f"Early stopping на эпохе {epoch+1}")
                    break
        else:
            # Если нет валидационного набора, сохраняем модель в конце каждой эпохи
            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_val_loss": train_loss,  # Используем train_loss вместо val_loss
                "train_loss": train_loss,
            }

            save_path_epoch = save_path.replace(".pth", f"_epoch_{epoch+1}.pth")
            torch.save(checkpoint, save_path_epoch)
            
            # Логируем модель в WandB
            if use_wandb:
                wandb.save(save_path_epoch)

    # Загружаем лучшую модель перед возвращением
    if os.path.exists(save_path):
        checkpoint = torch.load(save_path)
        model.load_state_dict(checkpoint["model_state_dict"])
        print(
            f"Загружена лучшая модель с валидационной потерей: {checkpoint['best_val_loss']:.4f}"
        )
    
    # Завершаем сессию WandB
    if use_wandb:
        wandb.finish()

    return model