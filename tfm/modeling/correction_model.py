import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import warnings

# Try importing Mamba; handle potential ImportError
try:
    from mamba_ssm import Mamba
except ImportError:
    Mamba = None
    raise ImportError(
        "Mamba is not installed. Please install it with `pip install mamba-ssm --no-build....`."
    )


class STMTKF(nn.Module):
    def __init__(
        self,
        config=None,
        input_type="mask",
        lambda_kf=0.1,
        mamba_dim=256,
        token_dim=256,
        num_tokens=8,
        state_dim=8,
        hidden_dim=128,
        device=None,
    ):
        super().__init__()

        # Проверка CUDA для Mamba
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            if not torch.cuda.is_available():
                warnings.warn(
                    "CUDA is not available, but Mamba requires CUDA to run efficiently. "
                    "The model may fail or run extremely slow on CPU."
                )
        else:
            self.device = device

        # Если есть конфиг, используем его параметры вместо аргументов по умолчанию
        if config is not None:
            self.input_type = config.get("input_type", input_type)
            self.lambda_kf = config.get("lambda_kf", lambda_kf)
            self.mamba_dim = config.get("mamba_dim", mamba_dim)
            self.token_dim = config.get("token_dim", token_dim)
            self.num_tokens = config.get("num_tokens", num_tokens)
            self.state_dim = config.get("state_dim", state_dim)
            self.hidden_dim = config.get("hidden_dim", hidden_dim)
        else:
            self.input_type = input_type
            self.lambda_kf = lambda_kf
            self.mamba_dim = mamba_dim
            self.token_dim = token_dim
            self.num_tokens = num_tokens
            self.state_dim = state_dim
            self.hidden_dim = hidden_dim

        # 3D-CNN для пространственно-временных паттернов
        self.conv3d = nn.Sequential(
            nn.Conv3d(4, 64, kernel_size=(3, 3, 3), padding=1),
            nn.ReLU(),
            nn.MaxPool3d((1, 2, 2)),
            nn.Conv3d(64, self.hidden_dim, kernel_size=(3, 3, 3), padding=1),
            nn.ReLU(),
        )

        # Mamba SSM для временных зависимостей
        self.mamba_proj = nn.Linear(self.hidden_dim, self.mamba_dim)
        self.mamba = Mamba(d_model=self.mamba_dim, d_state=16, d_conv=4, expand=2)

        # Проекция из mamba_dim в token_dim
        self.mamba_to_token = nn.Linear(self.mamba_dim, self.token_dim)

        # Генератор токенов движения
        self.q_proj = nn.Linear(self.token_dim, self.token_dim)
        self.k_proj = nn.Linear(self.token_dim, self.token_dim)
        self.v_proj = nn.Linear(self.token_dim, self.token_dim)
        self.learned_tokens = nn.Parameter(torch.randn(self.num_tokens, self.token_dim))

        # Обучаемые параметры Kalman Filter
        self.kf_fc = nn.Linear(self.token_dim, self.state_dim * self.state_dim)

        # Создаем фиксированные матрицы для KF
        self.register_buffer("F_base", torch.eye(self.state_dim, dtype=torch.float16))
        self.register_buffer("H", torch.zeros(4, self.state_dim, dtype=torch.float16))
        # Первые 4 компонента состояния соответствуют x, y, w, h
        self.H[:, :4] = torch.eye(4, dtype=torch.float16)

        # Рефинер масок/bbox
        self.mask_refiner = nn.Sequential(
            nn.Conv2d(self.hidden_dim + self.state_dim, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 1, kernel_size=1),
            nn.Sigmoid(),  # Добавляем сигмоиду для получения значений в диапазоне [0, 1]
        )

        # Перемещаем модель на нужное устройство
        self.to(self.device)

    @classmethod
    def from_config(cls, config, device=None):
        return cls(config=config, device=device)

    def attention(self, q, k, v):
        # q: [B, n_q, dim], k: [B, n_k, dim], v: [B, n_k, dim]
        # Вычисляем скалярное произведение и применяем softmax
        attn_scores = torch.bmm(q, k.transpose(1, 2)) / (self.token_dim**0.5)
        attn_weights = F.softmax(attn_scores, dim=-1)
        # Применяем веса внимания к значениям
        return torch.bmm(attn_weights, v)  # [B, n_q, dim]

    def forward(self, frames: torch.Tensor, inputs: torch.Tensor) -> torch.Tensor:
        """
        Args:
            frames: [B, N, 3, H, W] - видеокадры
            inputs: [B, N, H, W] для масок или [B, N, 4] для bbox
        Returns:
            [B, 1, H/2, W/2] (mask) или [B, 4] (bbox)
        """
        # Get model's native dtype and use it consistently
        model_dtype = next(self.parameters()).dtype
        
        # Ensure inputs are on the correct device and dtype
        frames = frames.to(device=self.device, dtype=model_dtype)
        inputs = inputs.to(device=self.device, dtype=model_dtype)

        B, N, C_frames, H, W = frames.shape

        # Преобразуем H и W в int для использования в torch.arange
        H_int = int(H)
        W_int = int(W)

        # Конвертация bbox → mask, если нужно
        if self.input_type == "bbox":
            masks = torch.zeros(B, N, H_int, W_int, device=self.device, dtype=model_dtype)
            for b in range(B):
                for n in range(N):
                    x, y, w, h = inputs[b, n]
                    x1, y1 = max(0, int((x - w / 2) * W_int)), max(
                        0, int((y - h / 2) * H_int)
                    )
                    x2, y2 = min(W_int - 1, int((x + w / 2) * W_int)), min(
                        H_int - 1, int((y + h / 2) * H_int)
                    )
                    if x1 < x2 and y1 < y2:  # Проверяем валидность bbox
                        masks[b, n, y1:y2, x1:x2] = 1
            inputs_processed = masks.unsqueeze(2)  # [B, N, 1, H, W]
        else:
            # Для масок добавляем канальное измерение
            inputs_processed = inputs.unsqueeze(2)  # [B, N, 1, H, W]

        # Объединение кадров и масок
        combined_input = torch.cat([frames, inputs_processed], dim=2)  # [B, N, 4, H, W]
        print(f"Combined input dtype: {combined_input.dtype}")
        # 3D-CNN: [B, N, 4, H, W] -> [B, hidden_dim, N, H/2, W/2]
        cnn_output = self.conv3d(combined_input.permute(0, 2, 1, 3, 4))

        # Получаем новые размеры после CNN
        _, C_hidden, N_out, H_out, W_out = cnn_output.shape

        # Преобразуем для Mamba: [B, hidden_dim, N, H/2, W/2] -> [B, N*H/2*W/2, hidden_dim]
        x_flat = cnn_output.permute(0, 2, 3, 4, 1).reshape(
            B, N_out * H_out * W_out, C_hidden
        )

        # Применяем Mamba
        x_mamba = self.mamba_proj(x_flat)  # [B, seq_len, mamba_dim]
        x_mamba = self.mamba(x_mamba)  # [B, seq_len, mamba_dim]

        # Проекция из mamba_dim в token_dim
        x_token = self.mamba_to_token(x_mamba)  # [B, seq_len, token_dim]

        # Применяем механизм внимания
        q = self.q_proj(self.learned_tokens).expand(
            B, -1, -1
        )  # [B, num_tokens, token_dim]
        k = self.k_proj(x_token)  # [B, seq_len, token_dim]
        v = self.v_proj(x_token)  # [B, seq_len, token_dim]
        motion_tokens = self.attention(q, k, v)  # [B, num_tokens, token_dim]

        # Генерация параметров матрицы перехода F
        delta_F = self.kf_fc(motion_tokens[:, 0]).view(
            B, self.state_dim, self.state_dim
        )
        F = self.F_base.unsqueeze(0) + delta_F  # [B, state_dim, state_dim]

        # Kalman Filter - инициализируем состояние
        state = torch.zeros(B, self.state_dim, device=self.device, dtype=model_dtype)  # [B, state_dim]

        # Подготовка матрицы H для batch операций
        H_batch = self.H.unsqueeze(0).expand(B, -1, -1)  # [B, 4, state_dim]
        H_t_batch = (
            self.H.transpose(0, 1).unsqueeze(0).expand(B, -1, -1)
        )  # [B, state_dim, 4]

        # Применяем фильтр Калмана для каждого кадра
        for t in range(N):
            # Предсказание: state = F @ state
            state = torch.bmm(F, state.unsqueeze(-1)).squeeze(-1)  # [B, state_dim]

            # Получение измерения
            if self.input_type == "bbox":
                measurement = inputs[:, t]  # [B, 4]
            else:
                # Для маски вычисляем bbox
                mask = inputs[:, t]  # [B, H, W]
                measurement = torch.zeros(B, 4, device=self.device, dtype=model_dtype)

                for b in range(B):
                    # Находим ненулевые элементы маски
                    non_zero = torch.nonzero(mask[b] > 0.5)
                    if len(non_zero) > 0:
                        # Вычисляем bbox
                        y_min, x_min = non_zero.min(dim=0)[0]
                        y_max, x_max = non_zero.max(dim=0)[0]

                        # Конвертируем в формат центр-ширина-высота
                        x = ((x_min + x_max) / 2) / W_int
                        y = ((y_min + y_max) / 2) / H_int
                        w = (x_max - x_min + 1) / W_int
                        h = (y_max - y_min + 1) / H_int

                        measurement[b] = torch.tensor([x, y, w, h], device=self.device, dtype=model_dtype)

            # Коррекция: state += K * (measurement - H @ state)
            z_pred = torch.bmm(H_batch, state.unsqueeze(-1)).squeeze(-1)  # [B, 4]
            innovation = measurement - z_pred
            K_innovation = torch.bmm(H_t_batch, innovation.unsqueeze(-1)).squeeze(
                -1
            )  # [B, state_dim]
            state = state + self.lambda_kf * K_innovation

        # Финальная обработка в зависимости от режима
        if self.input_type == "mask":
            # Извлекаем последние временные срезы из CNN выхода
            last_frame_features = cnn_output[:, :, -1]  # [B, hidden_dim, H/2, W/2]

            # Расширяем состояние KF для конкатенации с features
            state_expanded = (
                state.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, H_out, W_out)
            )

            # Конкатенируем features и state_expanded по измерению каналов
            combined = torch.cat(
                [last_frame_features, state_expanded], dim=1
            )  # [B, hidden_dim+state_dim, H/2, W/2]

            # Генерируем финальную маску
            return self.mask_refiner(combined)  # [B, 1, H/2, W/2]
        else:
            # Возвращаем предсказанные bbox координаты
            return state[:, :4]  # [B, 4]


# Пример использования для режима "mask"
def example_mask_mode():
    # Создание модели из конфига
    config = {
        "input_type": "mask",
        "lambda_kf": 0.2,
        "mamba_dim": 512,
        "token_dim": 256,
        "num_tokens": 16,
        "state_dim": 8,
        "hidden_dim": 128,
    }

    model = STMTKF.from_config(config)
    model.eval()  # Переводим в режим оценки

    # Создаем синтетические данные
    # Допустим, у нас есть 1 пример в батче, 5 кадров видео, размер кадра 128x128
    batch_size, seq_len, height, width = 1, 5, 128, 128

    # Видеокадры: [B, N, 3, H, W]
    frames = torch.randn(batch_size, seq_len, 3, height, width)

    # Маски: [B, N, H, W]
    masks = torch.zeros(batch_size, seq_len, height, width)

    # Создаем простую движущуюся маску - квадрат, перемещающийся по диагонали
    for t in range(seq_len):
        x1, y1 = 20 + t * 10, 20 + t * 10
        x2, y2 = x1 + 30, y1 + 30
        masks[0, t, y1:y2, x1:x2] = 1.0

    # Пропускаем через модель
    with torch.no_grad():
        refined_mask = model(frames, masks)

    print(f"Refined mask shape: {refined_mask.shape}")
    return frames, masks, refined_mask


# Пример использования для режима "bbox"
def example_bbox_mode():
    # Создание модели из конфига
    config = {
        "input_type": "bbox",
        "lambda_kf": 0.2,
        "mamba_dim": 32,
        "token_dim": 64,
        "num_tokens": 8,
        "state_dim": 4,
        "hidden_dim": 64,
    }

    model = STMTKF.from_config(config)
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {trainable_params / 1e6}M")
    model.eval()
    # Создаем синтетические данные
    batch_size, seq_len, height, width = 1, 4, 128, 128

    # Видеокадры: [B, N, 3, H, W]
    frames = torch.randn(batch_size, seq_len, 3, height, width)

    # Bounding boxes: [B, N, 4] где каждый bbox задан как [x, y, w, h] в нормализованных координатах
    bboxes = torch.zeros(batch_size, seq_len, 4)

    # Создаем движущийся bbox
    for t in range(seq_len):
        # Координаты центра
        x = 0.3 + t * 0.05
        y = 0.3 + t * 0.05
        # Ширина и высота
        w, h = 0.2, 0.2
        bboxes[0, t] = torch.tensor([x, y, w, h])

    # Пропускаем через модель
    with torch.no_grad():
        predicted_bbox = model(frames, bboxes)

    print(f"Predicted bbox: {predicted_bbox}")
    return frames, bboxes, predicted_bbox


# Визуализация результатов для примера с масками
def visualize_mask_results(frames, masks, refined_mask):
    import matplotlib.pyplot as plt
    import numpy as np

    # Получаем размеры из входных тензоров
    B, N, C, height, width = frames.shape

    plt.figure(figsize=(15, 10))

    # Отображаем последний кадр
    plt.subplot(2, 2, 1)
    frame = frames[0, -1].permute(1, 2, 0).cpu().numpy()
    # Нормализуем значения, если они выходят за пределы [0, 1]
    if frame.max() > 1.0:
        frame = frame / 255.0
    plt.imshow(np.clip(frame, 0, 1))
    plt.title("Последний кадр")
    plt.axis("off")

    # Отображаем маску на последнем кадре
    plt.subplot(2, 2, 2)
    mask = masks[0, -1].cpu().numpy()
    plt.imshow(mask, cmap="jet", alpha=0.7)
    plt.title("Исходная маска")
    plt.axis("off")

    # Отображаем уточненную маску
    plt.subplot(2, 2, 3)
    # Если refined_mask имеет форму [B, 1, H/2, W/2], то upsample её до [B, H, W]
    if refined_mask.shape[-2:] != (height, width):
        refined_mask_upsampled = torch.nn.functional.interpolate(
            refined_mask, size=(height, width), mode="nearest"
        )
        refined = refined_mask_upsampled[0, 0].cpu().numpy()
    else:
        refined = refined_mask[0, 0].cpu().numpy()
    plt.imshow(refined, cmap="jet", alpha=0.7)
    plt.title("Уточненная маска")
    plt.axis("off")

    # Отображаем наложение уточненной маски на кадр
    plt.subplot(2, 2, 4)
    plt.imshow(np.clip(frame, 0, 1))
    plt.imshow(refined, cmap="jet", alpha=0.5)
    plt.title("Наложение уточненной маски")
    plt.axis("off")

    plt.tight_layout()
    plt.show()


# Визуализация результатов для режима bbox
def visualize_bbox_results(frames, input_bboxes, predicted_bbox):
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    import numpy as np

    # Получаем размеры из входных тензоров
    B, N, C, height, width = frames.shape

    # Визуализируем последний кадр с input и predicted bbox
    plt.figure(figsize=(10, 10))

    # Отображаем последний кадр
    frame = frames[0, -1].permute(1, 2, 0).cpu().numpy()
    if frame.max() > 1.0:
        frame = frame / 255.0
    plt.imshow(np.clip(frame, 0, 1))

    # Отображаем input bbox на последнем кадре (красный)
    x, y, w, h = input_bboxes[0, -1].cpu().numpy()
    x1, y1 = int((x - w / 2) * width), int((y - h / 2) * height)
    rect_width, rect_height = int(w * width), int(h * height)
    rect = patches.Rectangle(
        (x1, y1), rect_width, rect_height, linewidth=2, edgecolor="r", facecolor="none"
    )
    plt.gca().add_patch(rect)

    # Отображаем predicted bbox (зеленый)
    x, y, w, h = predicted_bbox[0].cpu().numpy()
    x1, y1 = int((x - w / 2) * width), int((y - h / 2) * height)
    rect_width, rect_height = int(w * width), int(h * height)
    rect = patches.Rectangle(
        (x1, y1), rect_width, rect_height, linewidth=2, edgecolor="g", facecolor="none"
    )
    plt.gca().add_patch(rect)

    plt.title("Последний кадр с bbox (красный: вход, зеленый: предсказание)")
    plt.axis("off")
    plt.tight_layout()
    plt.show()


# Пример запуска
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    #print("Running mask mode example...")
   # frames, masks, refined_mask = example_mask_mode()
    #visualize_mask_results(frames, masks, refined_mask)

    print("\nRunning bbox mode example...")
    frames, bboxes, predicted_bbox = example_bbox_mode()
    visualize_bbox_results(frames, bboxes, predicted_bbox)
