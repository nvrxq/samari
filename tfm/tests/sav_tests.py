import random
import unittest
import os
from tfm.dataset.sav import AsyncAVDataset


class TestAsyncAVDataset(unittest.TestCase):
    def setUp(self):
        # Предполагаем, что тестовые данные находятся в директории data/sav
        self.test_dir = "tfm/tests/sav_example/"
        self.dataset = AsyncAVDataset(sav_dir=self.test_dir)

    def test_video_and_mask_loading(self):
        # Проверяем, что директория существует
        self.assertTrue(os.path.exists(self.test_dir), "Test directory does not exist")

        # Берем первый файл для теста
        test_file = self.dataset.files[0]

        # Проверяем загрузку видео
        frames = self.dataset._load_and_read_video(test_file)
        self.assertGreater(len(frames), 0, "Video should contain frames")
        print(f"Video length: {len(frames)} frames")

        # Проверяем загрузку маски
        mask = self.dataset._load_and_annot(test_file)
        self.assertIsNotNone(mask, "Mask should not be None")
        print(f"Mask content: {mask}")

        if self.binary_task:
            mask = random.choice(mask)


if __name__ == "__main__":
    unittest.main()
