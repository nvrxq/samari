import numpy as np
import logging
import cv2


logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def overlay_mask_on_frame(
    frame: np.ndarray, mask: np.ndarray, color: tuple = (255, 0, 0), alpha: float = 0.5
) -> np.ndarray:
    mask = (mask * 255).astype(np.uint8)
    mask = cv2.merge([mask, mask, mask])
    overlay = cv2.addWeighted(frame, 1, mask, alpha, 0)
    return overlay


def create_video_with_masks(dataset, output_path: str, video_index: int = 0):
    # Получаем данные из датасета
    frames, masks = dataset[video_index]

    # Проверяем корректность данных
    assert len(frames) == len(masks), "Mismatch between frames and masks count"
    if len(frames) == 0:
        logger.error("No frames found")
        return

    # Определяем параметры видео
    height, width, _ = frames[0].shape
    fps = 30.0  # Используем стандартный FPS
    fourcc = cv2.VideoWriter_fourcc(*"XVID")  # Измененный кодек

    # Создаем VideoWriter
    video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Проверка успешного открытия
    if not video_writer.isOpened():
        logger.error(f"Failed to open video writer for {output_path}")
        return

    try:
        for frame, mask in zip(frames, masks):
            # Преобразуем маску в бинарный формат
            mask = (mask > 0).astype(np.uint8)
            # plt.imshow(mask)
            # plt.show()

            # Накладываем маску
            frame_with_mask = overlay_mask_on_frame(frame, mask)

            # Конвертируем в BGR
            frame_bgr = cv2.cvtColor(frame_with_mask, cv2.COLOR_RGB2BGR)

            # Записываем кадр
            video_writer.write(frame_bgr)
    finally:
        video_writer.release()
        logger.info(f"Video saved to {output_path}")
