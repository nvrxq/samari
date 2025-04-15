import torch
import torch.nn as nn
import torch.nn.functional as F


class LearnableKalmanTracker(nn.Module):
    """
    Улучшенный обучаемый фильтр Калмана для tracking.
    Добавлены дропауты и другие механизмы регуляризации.
    """

    def __init__(
        self,
        state_dim=8,  # [x, y, w, h, vx, vy, vw, vh]
        meas_dim=4,  # [x, y, w, h]
        process_noise_scale=0.01,
        measurement_noise_scale=0.1,
        dropout_rate=0.1,
        use_adaptive_noise=True,
    ):
        super().__init__()

        self.state_dim = state_dim
        self.meas_dim = meas_dim
        self.use_adaptive_noise = use_adaptive_noise

        # Параметры матрицы перехода
        F_diag = torch.ones(state_dim)
        F_vel = torch.zeros(state_dim // 2)
        self.F_diag = nn.Parameter(F_diag)
        self.F_vel = nn.Parameter(F_vel)

        # Матрица наблюдения
        H_base = torch.zeros(meas_dim, state_dim)
        H_base[:meas_dim, :meas_dim] = torch.eye(meas_dim)
        self.H_base = nn.Parameter(H_base)

        # Параметры шума с дропаутом
        self.process_noise_scale = nn.Parameter(torch.tensor(process_noise_scale))
        self.measurement_noise_scale = nn.Parameter(
            torch.tensor(measurement_noise_scale)
        )

        # Весовой параметр для Калмановского обновления
        self.measurement_weight = nn.Parameter(torch.tensor(0.5))

        # Дропаут для регуляризации
        self.dropout = nn.Dropout(dropout_rate)

        # Улучшенная архитектура для адаптивного шума
        if use_adaptive_noise:
            self.adaptive_noise = nn.Sequential(
                nn.Linear(meas_dim * 2, 32),
                nn.LayerNorm(32),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
                nn.Linear(32, 16),
                nn.LayerNorm(16),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
                nn.Linear(16, state_dim),
                nn.Sigmoid(),  # Нормализация выхода
            )

        # Сеть для динамического выбора веса измерения
        self.confidence_estimator = nn.Sequential(
            nn.Linear(meas_dim * 2, 16),
            nn.LayerNorm(16),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(16, 1),
            nn.Sigmoid(),  # Выход в диапазоне [0, 1]
        )

        # Инициализация состояния
        self.reset()

    def reset(self):
        """Сбрасывает состояние трекера"""
        self.mean = None
        self.covariance = None
        self.last_measurement = None
        self.innovation_history = []
        self.training_mode = self.training

    def _build_F(self):
        """Строит матрицу перехода состояния"""
        batch_size = 1
        F = torch.zeros(
            batch_size, self.state_dim, self.state_dim, device=self.F_diag.device
        )

        # Заполняем диагональ
        F_diag_values = torch.clamp(
            self.F_diag, min=0.9, max=1.1
        )  # Ограничиваем значения
        for i in range(self.state_dim):
            F[:, i, i] = F_diag_values[i]

        # Заполняем смещения по скорости (значения над диагональю)
        half_dim = self.state_dim // 2
        for i in range(half_dim):
            F[:, i, i + half_dim] = torch.sigmoid(
                self.F_vel[i]
            )  # Сигмоида для стабильности

        return F

    def _build_Q(self, innovation=None):
        """
        Строит матрицу ковариации шума процесса.
        Может быть адаптивной в зависимости от ошибки измерения.
        """
        batch_size = 1
        base_scale = torch.exp(
            self.process_noise_scale
        )  # Экспонента для положительности
        Q = (
            torch.eye(self.state_dim, device=self.F_diag.device).expand(
                batch_size, -1, -1
            )
            * base_scale
        )

        # Если используем адаптивный шум и есть ошибка инновации
        if (
            self.use_adaptive_noise
            and innovation is not None
            and self.last_measurement is not None
        ):
            with torch.no_grad():
                input_features = torch.cat([innovation, self.last_measurement], dim=-1)
                adaptive_factors = self.adaptive_noise(input_features)

                # Применяем дропаут во время обучения для регуляризации
                if self.training:
                    adaptive_factors = self.dropout(adaptive_factors)

                # Масштабируем базовый шум
                adaptive_scale = adaptive_factors.view(batch_size, self.state_dim, 1)
                Q = Q * (1.0 + adaptive_scale)

        return Q

    def _build_R(self, measurement):
        """Строит матрицу ковариации шума измерения"""
        batch_size = measurement.shape[0]
        base_scale = torch.exp(
            self.measurement_noise_scale
        )  # Экспонента для положительности
        R = (
            torch.eye(self.meas_dim, device=measurement.device).expand(
                batch_size, -1, -1
            )
            * base_scale
        )
        return R

    def _build_H(self):
        """Строит матрицу наблюдения"""
        batch_size = 1
        H = self.H_base.expand(batch_size, -1, -1)
        return H

    def predict(self):
        """
        Предсказывает следующее состояние без обновления

        Returns:
            predicted_bbox: предсказанный бокс в формате [x, y, w, h]
        """
        if self.mean is None:
            raise RuntimeError(
                "Трекер не инициализирован. Сначала вызовите reset() и передайте первое измерение."
            )

        # Строим матрицу перехода
        F = self._build_F()

        # Предсказываем новое состояние
        predicted_mean = torch.bmm(F, self.mean.unsqueeze(-1)).squeeze(-1)

        # Предсказываем новую ковариацию
        Ft = F.transpose(1, 2)
        Q = self._build_Q()
        predicted_cov = torch.bmm(torch.bmm(F, self.covariance), Ft) + Q

        # Возвращаем только предсказанный бокс
        predicted_bbox = predicted_mean[:, : self.meas_dim]

        return predicted_bbox[0]  # Возвращаем первый (и единственный) бокс

    def update(self, measurement):
        """
        Обновляет состояние фильтра Калмана на основе измерения

        Args:
            measurement: текущее измерение бокса [x, y, w, h]

        Returns:
            updated_bbox: обновленный бокс в формате [x, y, w, h]
        """
        if self.mean is None:
            raise RuntimeError(
                "Трекер не инициализирован. Сначала вызовите reset() и передайте первое измерение."
            )

        # Предсказание
        F = self._build_F()
        predicted_mean = torch.bmm(F, self.mean.unsqueeze(-1)).squeeze(-1)
        Ft = F.transpose(1, 2)
        Q = self._build_Q()
        predicted_cov = torch.bmm(torch.bmm(F, self.covariance), Ft) + Q

        # Подготовка измерения
        if not isinstance(measurement, torch.Tensor):
            measurement = torch.tensor(measurement, device=self.F_diag.device).float()
        if len(measurement.shape) == 1:
            measurement = measurement.unsqueeze(0)

        # Инновация (ошибка измерения)
        H = self._build_H()
        Ht = H.transpose(1, 2)
        R = self._build_R(measurement)

        innovation = measurement - torch.bmm(H, predicted_mean.unsqueeze(-1)).squeeze(
            -1
        )

        # Для адаптивного шума сохраняем инновацию
        if self.use_adaptive_noise:
            self.innovation_history.append(innovation)
            if len(self.innovation_history) > 5:  # Ограничиваем историю
                self.innovation_history.pop(0)

        # Коэффициент Калмана с учетом уверенности в измерении
        S = torch.bmm(torch.bmm(H, predicted_cov), Ht) + R
        K = torch.bmm(torch.bmm(predicted_cov, Ht), torch.inverse(S))

        # Динамически определяем вес измерения, если есть предыдущее измерение
        measurement_weight = torch.sigmoid(self.measurement_weight)
        if self.last_measurement is not None:
            input_features = torch.cat([innovation, measurement], dim=-1)
            confidence = self.confidence_estimator(input_features)
            measurement_weight = confidence * measurement_weight

        # Обновление состояния с взвешенным влиянием измерения
        self.mean = predicted_mean + measurement_weight * torch.bmm(
            K, innovation.unsqueeze(-1)
        ).squeeze(-1)
        I = torch.eye(self.state_dim, device=K.device).expand_as(predicted_cov)
        self.covariance = torch.bmm((I - torch.bmm(K, H)), predicted_cov)

        # Сохраняем текущее измерение
        self.last_measurement = measurement

        # Возвращаем обновленный бокс
        updated_bbox = self.mean[:, : self.meas_dim]

        return updated_bbox[0]  # Возвращаем первый (и единственный) бокс

    def forward(self, measurement):
        """
        Обрабатывает измерение, возвращает обновленное состояние

        Args:
            measurement: текущее измерение бокса [x, y, w, h]

        Returns:
            bbox: предсказанный бокс в формате [x, y, w, h]
        """
        if self.mean is None:
            # Инициализация состояния при первом измерении
            if not isinstance(measurement, torch.Tensor):
                measurement = torch.tensor(
                    measurement, device=self.F_diag.device
                ).float()
            if len(measurement.shape) == 1:
                measurement = measurement.unsqueeze(0)

            batch_size = measurement.shape[0]
            self.mean = torch.zeros(
                batch_size, self.state_dim, device=measurement.device
            )
            self.mean[:, : self.meas_dim] = measurement

            self.covariance = torch.eye(
                self.state_dim, device=measurement.device
            ).expand(batch_size, -1, -1)
            self.last_measurement = measurement

            return measurement[0]  # Возвращаем первое измерение при инициализации
        else:
            # Штатный режим: предсказание и обновление
            return self.update(measurement)


# Пример использования
if __name__ == "__main__":
    from samari_loss import KalmanLoss

    BS, SEQ_LEN, DIM = 4, 1, 4
    model = LearnableKalmanTracker()

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {trainable_params}")
    dummy_input = torch.randn(BS, SEQ_LEN, DIM)
    dummy_target = torch.randn(BS, SEQ_LEN, DIM)
    loss_fn = KalmanLoss()
    loss = loss_fn(dummy_input, dummy_target)
    print(loss)
