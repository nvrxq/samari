import torch
import torchvision.transforms as transforms
import cv2
import numpy as np
from PIL import Image
import os
import glob
import argparse
from tqdm import tqdm

# Предполагается, что эти модули находятся в вашем проекте
from tfm.modeling.model import TemporalFusionModule, config_tiny_mamba_bbox
# Если вы используете LightningModule для загрузки, раскомментируйте:
from tfm.trainer import TemporalFusionTrainer

# --- Константы и Настройки ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Важно: Используйте те же трансформации, что и при обучении!
# Пример (адаптируйте под ваши реальные трансформации):
IMG_TRANSFORM = transforms.Compose([
    transforms.Resize((64, 64)), # Пример размера, используйте ваш размер
    transforms.ToTensor(),
    # Используйте ту же нормализацию, что и при обучении
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
# Цвет для отрисовки BBox (BGR)
BBOX_COLOR = (0, 255, 0) # Зеленый
BBOX_THICKNESS = 2

# --- Вспомогательные функции ---

def load_frames(input_dir):
    """Загружает и сортирует кадры из директории."""
    img_paths = sorted(glob.glob(os.path.join(input_dir, '*.jpg'))) # Или .png, .jpeg и т.д.
    if not img_paths:
        raise FileNotFoundError(f"Не найдено изображений в {input_dir}")
    frames = [Image.open(p).convert('RGB') for p in img_paths]
    return frames, img_paths

def normalize_bbox(bbox, img_w, img_h):
    """Нормализует bbox [x, y, w, h] к [0, 1]."""
    x, y, w, h = bbox
    norm_x = x / img_w
    norm_y = y / img_h
    norm_w = w / img_w
    norm_h = h / img_h
    return torch.tensor([norm_x, norm_y, norm_w, norm_h], dtype=torch.float32).clamp(0.0, 1.0)

def denormalize_bbox(norm_bbox, img_w, img_h):
    """Преобразует нормализованный bbox [norm_x, norm_y, norm_w, norm_h] обратно в пиксельные координаты [x, y, w, h]."""
    norm_x, norm_y, norm_w, norm_h = norm_bbox
    x = norm_x * img_w
    y = norm_y * img_h
    w = norm_w * img_w
    h = norm_h * img_h
    return np.array([x, y, w, h])

def draw_bbox(image_np, bbox_pixel):
    """Рисует bbox [x, y, w, h] на изображении (numpy array BGR)."""
    x, y, w, h = map(int, bbox_pixel)
    x_min, y_min = x, y
    x_max, y_max = x + w, y + h
    cv2.rectangle(image_np, (x_min, y_min), (x_max, y_max), BBOX_COLOR, BBOX_THICKNESS)
    return image_np

# --- Основная функция инференса ---

def run_inference(checkpoint_path, input_dir, output_video_path, initial_bbox_str):
    """
    Запускает инференс модели на последовательности кадров.

    Args:
        checkpoint_path (str): Путь к файлу чекпоинта модели (.ckpt).
        input_dir (str): Директория с кадрами видео (изображениями).
        output_video_path (str): Путь для сохранения выходного видео.
        initial_bbox_str (str): Строка с начальным bbox "x,y,w,h" для первого кадра.
    """
    print(f"Используется устройство: {DEVICE}")
    print("Загрузка модели...")

    # --- Загрузка модели ---
    # Вариант 1: Если чекпоинт сохранялся напрямую из TemporalFusionModule
    model = TemporalFusionModule(config_tiny_mamba_bbox)
    # Загружаем state_dict. Нужно убедиться, что ключи совпадают.
    # Если чекпоинт от Lightning, ключи могут иметь префикс "model."
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    # Адаптируем state_dict, если он от Lightning (удаляем префикс 'model.')
    state_dict = checkpoint.get('state_dict', checkpoint) # Получаем state_dict
    model_state_dict = {k.replace('model.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(model_state_dict)

    # Вариант 2: Если чекпоинт сохранялся через LightningModule
    # try:
    #     # Загружаем LightningModule, затем получаем саму модель
    #     pl_module = TemporalFusionTrainer.load_from_checkpoint(
    #         checkpoint_path,
    #         map_location='cpu', # Загружаем на CPU сначала
    #         config=config_tiny_mamba_bbox # Передаем конфиг, если он нужен в __init__ трейнера
    #     )
    #     model = pl_module.model
    # except TypeError as e:
    #      print(f"Ошибка при загрузке LightningModule: {e}")
    #      print("Попытка загрузить state_dict напрямую (Вариант 1)...")
    #      model = TemporalFusionModule(config_tiny_mamba_bbox)
    #      checkpoint = torch.load(checkpoint_path, map_location='cpu')
    #      state_dict = checkpoint.get('state_dict', checkpoint)
    #      model_state_dict = {k.replace('model.', ''): v for k, v in state_dict.items()}
    #      model.load_state_dict(model_state_dict)


    model.to(DEVICE)
    model.eval() # Переводим модель в режим инференса

    print("Загрузка кадров...")
    # --- Загрузка и подготовка данных ---
    pil_frames, frame_paths = load_frames(input_dir)
    num_frames = len(pil_frames)
    if num_frames < 2:
        raise ValueError("Требуется как минимум 2 кадра для инференса.")

    # Получаем размеры из первого кадра
    img_w, img_h = pil_frames[0].size

    # Парсим и нормализуем начальный bbox
    try:
        initial_bbox_pixel = list(map(float, initial_bbox_str.split(',')))
        if len(initial_bbox_pixel) != 4: raise ValueError
    except:
        raise ValueError("Неверный формат initial_bbox. Ожидается 'x,y,w,h'.")

    initial_bbox_norm = normalize_bbox(initial_bbox_pixel, img_w, img_h)
    # Добавляем batch dimension и переносим на устройство
    # Shape: (1, 1, 4) - batch=1, seq_len=1, coords=4
    past_boxes_norm = initial_bbox_norm.unsqueeze(0).unsqueeze(0).to(DEVICE)

    # Список для хранения предсказанных *пиксельных* bbox [x, y, w, h]
    predicted_bboxes_pixel = [np.array(initial_bbox_pixel)]

    print("Запуск инференса...")
    # --- Цикл инференса ---
    # Обрабатываем кадры начиная со второго (индекс 1)
    # Для предсказания кадра `t` нам нужны кадры `0...t` и боксы `0...t-1`
    output_frames_cv = [] # Для сохранения кадров с отрисованными боксами

    # Обрабатываем первый кадр (добавляем GT бокс)
    frame0_cv = cv2.cvtColor(np.array(pil_frames[0]), cv2.COLOR_RGB2BGR)
    frame0_with_bbox = draw_bbox(frame0_cv.copy(), initial_bbox_pixel)
    output_frames_cv.append(frame0_with_bbox)


    with torch.no_grad():
        for t in tqdm(range(1, num_frames), desc="Инференс"):
            # 1. Подготовка кадров: берем все кадры до текущего включительно
            # Список PIL Image -> Список Tensor [C, H, W]
            current_pil_frames = pil_frames[:t+1]
            input_frames_tensor = torch.stack([IMG_TRANSFORM(frame) for frame in current_pil_frames])
            # Добавляем batch dimension: [T, C, H, W] -> [1, T, C, H, W]
            input_frames_batch = input_frames_tensor.unsqueeze(0).to(DEVICE)
            # T на данном шаге равно t+1

            # 2. Подготовка прошлых боксов: `past_boxes_norm` уже содержит боксы 0...t-1
            # Shape: (1, t, 4)

            # 3. Запуск модели
            # Вход: frames (1, t+1, C, H, W), past_bboxes (1, t, 4)
            # Выход: predicted_bboxes (1, t+1, 4)
            predicted_bboxes_norm_all = model(input_frames_batch, past_boxes_norm)

            # 4. Извлечение нужного предсказания
            # Нам нужен бокс для *последнего* кадра в последовательности (кадр t)
            # Он находится в конце выхода: predicted_bboxes_norm_all[0, -1, :]
            predicted_bbox_norm_t = predicted_bboxes_norm_all[0, -1, :].cpu() # Shape: (4,)

            # 5. Денормализация и сохранение
            predicted_bbox_pixel_t = denormalize_bbox(predicted_bbox_norm_t.numpy(), img_w, img_h)
            predicted_bboxes_pixel.append(predicted_bbox_pixel_t)

            # 6. Обновление `past_boxes_norm` для следующей итерации
            # Добавляем предсказанный *нормализованный* бокс
            # Добавляем batch и seq dimension: (4,) -> (1, 1, 4)
            new_box_to_add = predicted_bbox_norm_t.unsqueeze(0).unsqueeze(0).to(DEVICE)
            # Конкатенируем: (1, t, 4) + (1, 1, 4) -> (1, t+1, 4)
            past_boxes_norm = torch.cat([past_boxes_norm, new_box_to_add], dim=1)

            # 7. Отрисовка бокса на текущем кадре (для видео)
            current_frame_cv = cv2.cvtColor(np.array(pil_frames[t]), cv2.COLOR_RGB2BGR)
            frame_with_bbox = draw_bbox(current_frame_cv.copy(), predicted_bbox_pixel_t)
            output_frames_cv.append(frame_with_bbox)


    # --- Сохранение результата ---
    print(f"Сохранение видео в {output_video_path}...")
    if not output_frames_cv:
        print("Нет кадров для сохранения.")
        return

    # Получаем размер кадра из первого обработанного кадра
    height, width, layers = output_frames_cv[0].shape
    size = (width, height)

    # Используем MP4V кодек, можно попробовать другие ('XVID', 'MJPG')
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_video = cv2.VideoWriter(output_video_path, fourcc, 25, size) # 25 FPS - можно настроить

    for frame in output_frames_cv:
        out_video.write(frame)

    out_video.release()
    print("Инференс завершен.")


# --- Запуск скрипта ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Инференс модели TemporalFusionModule для трекинга.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Путь к файлу чекпоинта (.ckpt).")
    parser.add_argument("--input_dir", type=str, required=True, help="Директория с кадрами видео (jpg, png).")
    parser.add_argument("--output_video", type=str, default="output_tracking.mp4", help="Путь для сохранения выходного видео.")
    parser.add_argument("--initial_bbox", type=str, required=True, help="Начальный bounding box для первого кадра в формате 'x,y,w,h' (пиксельные координаты).")

    args = parser.parse_args()

    run_inference(args.checkpoint, args.input_dir, args.output_video, args.initial_bbox)

    # Пример запуска:
    # python inference.py --checkpoint path/to/your/model.ckpt --input_dir path/to/frames/ --initial_bbox "150,200,50,80" --output_video tracked_video.mp4