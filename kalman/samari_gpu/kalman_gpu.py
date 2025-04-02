import torch
from kalman_filter import KalmanFilter

# Инициализация фильтра
kf = KalmanFilter()

# Батч из двух измерений (формат: [x, y, aspect_ratio, height])
measurements = torch.tensor(
    [[50.0, 50.0, 1.0, 30.0], [100.0, 100.0, 0.8, 25.0]], device="cuda"
)

# Инициализация треков
mean, cov = kf.initiate(measurements)
print("Initial mean:\n", mean.cpu())
print("Initial cov:\n", cov[0].cpu())

# Предсказание
mean_pred, cov_pred = kf.predict(mean, cov)
print("\nPredicted mean:\n", mean_pred.cpu())

# Проекция
proj_mean, proj_cov = kf.project(mean_pred, cov_pred)
print("\nProjected mean:\n", proj_mean.cpu())
