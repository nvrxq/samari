from torch.utils.data import Dataset
from random import choice
import cv2
import numpy as np

# from utils import add_gaussian_noise, simulate_occlusions, add_outliers
import glob
import os
import logging
import random
import torch

logger = logging.getLogger(__name__)


class SavSamariDataset(Dataset):
    def __init__(
        self,
        root_dir,
        max_frames_per_sample=64,
        add_noise=True,
        noise_level=0.05,
        random_occlusions=True,
        occlusion_probability=0.1,
        steps_per_epoch: int = 100,
        min_seq_len: int = 20,
        max_occlusion_length=5,
        outlier_prob=0.05,
        outlier_scale=3.0,
    ):
        self.root_dir = root_dir
        self.max_frames_per_sample = max_frames_per_sample
        self.add_noise = add_noise
        self.noise_level = noise_level
        self.random_occlusions = random_occlusions
        self.occlusion_probability = occlusion_probability
        self.max_occlusion_length = max_occlusion_length
        self.steps_per_epoch = steps_per_epoch
        self.min_seq_len = min_seq_len
        self.outlier_prob = outlier_prob
        self.outlier_scale = outlier_scale

        self.files = glob.glob(os.path.join(root_dir, "*"))
        self.dir2labels = {}
        self._create_labels()

        logger.info(
            f"Your dataset setup:\n"
            f"Root dir: {self.root_dir}\n"
            f"Max frames per sample: {self.max_frames_per_sample}\n"
            f"Add noise: {self.add_noise}\n"
            f"Noise level: {self.noise_level}\n"
            f"Random occlusions: {self.random_occlusions}\n"
            f"Occlusion probability: {self.occlusion_probability}\n"
            f"Max occlusion length: {self.max_occlusion_length}\n"
            f"Outlier probability: {self.outlier_prob}\n"
            f"Steps per epoch: {self.steps_per_epoch}"
        )

    def _create_labels(self):
        for dir_path in self.files:
            label_file = os.path.join(dir_path, "groundtruth.txt")
            imgs = sorted(glob.glob(os.path.join(dir_path, "img", "*.jpg")))
            self.dir2labels[dir_path] = {"label_file": label_file, "imgs": imgs}
        logger.info(f"Found {len(self.dir2labels)} directories")

    def load_labels(self, label_file):
        labels = []
        with open(label_file, "r") as f:
            for line in f:
                x, y, w, h = map(float, line.strip().split(","))
                labels.append((x, y, w, h))
        return labels

    def add_gaussian_noise(self, boxes):
        """
        Добавляет гауссовский шум к координатам и размерам боксов

        Args:
            boxes: список или тензор боксов формата (x, y, w, h)

        Returns:
            noisy_boxes: зашумленные боксы
        """
        if isinstance(boxes, list):
            boxes = torch.tensor(boxes)

        noisy_boxes = boxes.clone()

        # Вычисляем стандартное отклонение на основе размеров объектов
        widths = boxes[:, 2]
        heights = boxes[:, 3]
        avg_size = (widths + heights) / 2

        # Генерируем шум для каждой координаты
        noise_x = torch.randn_like(boxes[:, 0]) * avg_size * self.noise_level
        noise_y = torch.randn_like(boxes[:, 1]) * avg_size * self.noise_level
        noise_w = torch.randn_like(boxes[:, 2]) * self.noise_level
        noise_h = torch.randn_like(boxes[:, 3]) * self.noise_level

        # Добавляем шум
        noisy_boxes[:, 0] += noise_x
        noisy_boxes[:, 1] += noise_y
        noisy_boxes[:, 2] *= 1 + noise_w  # Мультипликативный шум для размеров
        noisy_boxes[:, 3] *= 1 + noise_h

        # Удостоверяемся, что размеры не стали отрицательными
        noisy_boxes[:, 2] = torch.clamp(noisy_boxes[:, 2], min=1.0)
        noisy_boxes[:, 3] = torch.clamp(noisy_boxes[:, 3], min=1.0)

        return noisy_boxes.tolist()

    def simulate_occlusions(self, boxes):
        """
        Симулирует окклюзии путем маркирования некоторых боксов как отсутствующие

        Args:
            boxes: список боксов формата (x, y, w, h)

        Returns:
            occluded_boxes: боксы с симулированными окклюзиями
            occlusion_mask: маска (1 = данные доступны, 0 = окклюзия)
        """
        seq_len = len(boxes)
        occluded_boxes = boxes.copy()
        occlusion_mask = [1] * seq_len

        t = 1  # Начинаем с 1-го кадра (0-й используем для инициализации)
        while t < seq_len:
            if random.random() < self.occlusion_probability:
                # Начало окклюзии
                occlusion_length = random.randint(
                    1, min(self.max_occlusion_length, seq_len - t)
                )
                for i in range(t, min(t + occlusion_length, seq_len)):
                    occlusion_mask[i] = 0
                t += occlusion_length
            else:
                t += 1

        return occluded_boxes, occlusion_mask

    def add_outliers(self, boxes):
        """
        Добавляет выбросы (резкие изменения положения), симулируя ошибки детектора

        Args:
            boxes: список боксов формата (x, y, w, h)

        Returns:
            boxes_with_outliers: боксы с добавленными выбросами
        """
        boxes_with_outliers = boxes.copy()  # Используем copy() для списков
        seq_len = len(boxes)

        # Для каждого кадра, кроме первого (первый используем для инициализации)
        for t in range(1, seq_len):
            if random.random() < self.outlier_prob:
                # Создаем выброс
                prev_box = boxes[t - 1]
                width, height = prev_box[2], prev_box[3]

                # Сильное смещение в случайном направлении
                shift_x = random.uniform(-1, 1) * width * self.outlier_scale
                shift_y = random.uniform(-1, 1) * height * self.outlier_scale

                # Применяем смещение
                boxes_with_outliers[t] = (
                    prev_box[0] + shift_x,
                    prev_box[1] + shift_y,
                    prev_box[2] * random.uniform(0.2, 2.0),  # Случайно меняем размер
                    prev_box[3] * random.uniform(0.2, 2.0),
                )

        return boxes_with_outliers

    def __len__(self):
        return self.steps_per_epoch

    def __getitem__(self, index):
        dir_path = np.random.choice(self.files)
        label_info = self.dir2labels[dir_path]
        raw_labels = self.load_labels(label_info["label_file"])

        # Проверяем, что последовательность достаточно длинная
        if len(raw_labels) < self.min_seq_len:
            # Если нет, рекурсивно вызываем метод снова с другим индексом
            return self.__getitem__(random.randint(0, len(self) - 1))

        max_start_index = len(raw_labels) - self.min_seq_len
        bos = np.random.randint(0, max_start_index + 1)
        eos = min(
            bos + np.random.randint(self.min_seq_len, self.max_frames_per_sample + 1),
            len(raw_labels),
        )
        selected_labels_xywh = raw_labels[bos:eos]

        # Создаем копию оригинальных боксов
        original_boxes = torch.tensor(selected_labels_xywh).float()

        # Применяем аугментации
        augmented_boxes = torch.tensor(selected_labels_xywh).float()
        occlusion_mask = None

        if self.add_noise:
            augmented_boxes = self.add_gaussian_noise(augmented_boxes)

        if self.random_occlusions:
            augmented_boxes, occlusion_mask = self.simulate_occlusions(augmented_boxes)

        augmented_boxes = self.add_outliers(augmented_boxes)

        # Формируем выходной словарь
        result = {
            "original_boxes": original_boxes,
            "boxes": torch.tensor(augmented_boxes).float(),
            "dir_path": dir_path,
            "seq_indices": list(range(bos, eos)),
        }

        if occlusion_mask is not None:
            result["occlusion_mask"] = torch.tensor(occlusion_mask).float()

        return result


def samari_collate_fn(batch):
    """
    Функция для объединения элементов датасета в батч с паддингом для последовательностей разной длины.
    
    Args:
        batch: список словарей от SavSamariDataset
        
    Returns:
        batched_sample: словарь с батчами тензоров
    """
    # Находим максимальную длину последовательности в батче
    max_seq_len = max(sample["original_boxes"].shape[0] for sample in batch)
    
    # Списки для сбора данных
    original_boxes_batch = []
    boxes_batch = []
    occlusion_masks_batch = []
    seq_lengths = []  # Сохраняем реальные длины последовательностей
    dir_paths = []
    seq_indices_batch = []
    
    # Обрабатываем каждый элемент в батче
    for sample in batch:
        # Получаем текущую длину последовательности
        seq_len = sample["original_boxes"].shape[0]
        seq_lengths.append(seq_len)
        
        # Оригинальные боксы: паддинг по времени до max_seq_len
        orig_boxes = sample["original_boxes"]
        padding_size = max_seq_len - seq_len
        if padding_size > 0:
            # Повторяем последний бокс для паддинга
            padding = orig_boxes[-1].unsqueeze(0).repeat(padding_size, 1)
            orig_boxes_padded = torch.cat([orig_boxes, padding], dim=0)
        else:
            orig_boxes_padded = orig_boxes
        original_boxes_batch.append(orig_boxes_padded)
        
        # Аугментированные боксы
        aug_boxes = sample["boxes"]
        if padding_size > 0:
            padding = aug_boxes[-1].unsqueeze(0).repeat(padding_size, 1)
            aug_boxes_padded = torch.cat([aug_boxes, padding], dim=0)
        else:
            aug_boxes_padded = aug_boxes
        boxes_batch.append(aug_boxes_padded)
        
        # Маска окклюзий (если есть)
        if "occlusion_mask" in sample:
            occ_mask = sample["occlusion_mask"]
            if padding_size > 0:
                # Для паддинга маски используем нули (считаем окклюзией)
                padding = torch.zeros(padding_size, device=occ_mask.device, dtype=occ_mask.dtype)
                occ_mask_padded = torch.cat([occ_mask, padding], dim=0)
            else:
                occ_mask_padded = occ_mask
            occlusion_masks_batch.append(occ_mask_padded)
        
        # Строковые и другие данные
        dir_paths.append(sample["dir_path"])
        
        # Индексы последовательности
        seq_indices = torch.tensor(sample["seq_indices"])
        if padding_size > 0:
            padding = torch.ones(padding_size, dtype=torch.long) * -1  # -1 для паддинга
            seq_indices_padded = torch.cat([seq_indices, padding], dim=0)
        else:
            seq_indices_padded = seq_indices
        seq_indices_batch.append(seq_indices_padded)
    
    # Объединяем данные в тензоры
    batched_sample = {
        "original_boxes": torch.stack(original_boxes_batch),
        "boxes": torch.stack(boxes_batch),
        "seq_lengths": torch.tensor(seq_lengths),
        "dir_paths": dir_paths,
        "seq_indices": torch.stack(seq_indices_batch)
    }
    
    # Добавляем маску окклюзий, если есть
    if occlusion_masks_batch:
        batched_sample["occlusion_mask"] = torch.stack(occlusion_masks_batch)
    
    # Создаем маску действительных (не паддинговых) элементов последовательности
    batch_size = len(batch)
    valid_mask = torch.zeros(batch_size, max_seq_len, dtype=torch.bool)
    for i, length in enumerate(seq_lengths):
        valid_mask[i, :length] = True
    batched_sample["valid_mask"] = valid_mask
    
    return batched_sample


if __name__ == "__main__":
    dataset = SavSamariDataset(
        root_dir="/home/never/Work/samari/datasets/lasot",
    )
    print(len(dataset))
    sample = dataset[0]
    print(sample["original_boxes"][-1])
    print(sample["boxes"][-1])
