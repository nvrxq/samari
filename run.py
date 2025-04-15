import torch
from torch.utils.data import DataLoader
from tfm.dataset.samset import SavSamariDataset, samari_collate_fn
from tfm.modeling.samari import LearnableKalmanTracker
from tfm.train import train_kalman_tracker

def main():
    # Параметры
    train_root = "/home/never/Work/samari/datasets/lasot"
    val_root = "/home/never/Work/samari/datasets/lasot_val"
    batch_size = 16
    num_epochs = 30
    learning_rate = 0.001
    weight_decay = 1e-5
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    save_path = "models/kalman_tracker.pth"
    load_checkpoint = None  # "models/previous_checkpoint.pth" если нужно продолжить обучение
    
    # Создаем датасеты
    train_dataset = SavSamariDataset(
        root_dir=train_root,
        max_frames_per_sample=32,
        add_noise=True,
        noise_level=0.05,
        random_occlusions=True,
        occlusion_probability=0.1,
        steps_per_epoch=1000,  # Количество батчей в эпохе
        min_seq_len=20,
        outlier_prob=0.2,
    )
    
    # Создаем валидационный датасет
    val_dataset = SavSamariDataset(
        root_dir=val_root,
        max_frames_per_sample=64,
        add_noise=True,
        noise_level=0.05,
        random_occlusions=True,
        occlusion_probability=0.1,
        steps_per_epoch=100,  # Меньше для валидации
        min_seq_len=20,
        outlier_prob=0.2
    )
    
    # Создаем загрузчики данных
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        collate_fn=samari_collate_fn
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        collate_fn=samari_collate_fn
    )
    
    # Создаем модель
    model = LearnableKalmanTracker(
        state_dim=8,
        meas_dim=4,
        process_noise_scale=0.01,
        measurement_noise_scale=0.1,
        dropout_rate=0.1,
        use_adaptive_noise=True
    )
    
    # Обучаем модель
    trained_model = train_kalman_tracker(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=num_epochs,
        lr=learning_rate,
        weight_decay=weight_decay,
        patience=5,
        device=device,
        save_path=save_path,
        load_checkpoint=load_checkpoint
    )
    
    # Сохраняем финальную модель
    torch.save(trained_model.state_dict(), "models/kalman_tracker_final.pth")
    print("Обучение завершено!")

if __name__ == "__main__":
    main()