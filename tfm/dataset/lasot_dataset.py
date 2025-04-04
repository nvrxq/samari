from torch.utils.data import Dataset
import torch
import glob
import os
import numpy as np
from PIL import Image


import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LaSotDataset(Dataset):
    def __init__(
        self, root_dir, transform=None, steps_per_epoch: int = 100, max_frames: int = 60
    ):
        self.root_dir = root_dir
        self.transform = transform
        self.files = glob.glob(root_dir + "/*")  # ["person-1", ...]
        self.dir2labels = {}
        self.max_frames = max_frames
        self.steps_per_epoch = steps_per_epoch
        self._create_labels()
        logger.info(
            f"Your dataset setup:\nRoot dir: {self.root_dir}\nTransform: {self.transform}\nSteps per epoch: {self.steps_per_epoch}\nMax frames: {self.max_frames}"
        )

    def _create_labels(self):
        for dir in self.files:
            label_file = os.path.join(dir, "groundtruth.txt")
            self.dir2labels[dir] = {
                "label_file": label_file,
                "imgs": glob.glob(os.path.join(dir + "/img", "*.jpg")),
            }
        logger.info(f"Found {len(self.dir2labels)} directories")

    def load_labels(self, label_file):
        labels = []
        with open(label_file, "r") as f:
            for line in f:
                x, y, w, h = map(float, line.split(","))
                labels.append((x, y, w, h))
        return labels

    def __len__(self):
        return self.steps_per_epoch

    def __getitem__(self, _):
        dir = np.random.choice(self.files)
        label_file = self.dir2labels[dir]["label_file"]
        img_paths = self.dir2labels[dir]["imgs"]
        # Sort image paths to ensure correct order
        img_paths.sort()

        # Load raw labels first
        raw_labels = self.load_labels(label_file)

        assert len(raw_labels) == len(img_paths), \
            f"Number of labels ({len(raw_labels)}) does not match number of images ({len(img_paths)}) in {dir}"

        # Select a sequence
        # Ensure sequence length is at least 2 and respects max_frames
        min_seq_len = 2 # Need at least 2 frames for KF logic (init + first step)
        if len(img_paths) < min_seq_len:
             # Handle cases with very few frames if necessary, e.g., skip or pad
             # For now, let's retry sampling (this might be inefficient)
             logger.warning(f"Skipping sequence from {dir}, too few frames ({len(img_paths)} < {min_seq_len})")
             return self.__getitem__(_) # Simple retry, consider better handling

        max_start_index = len(img_paths) - min_seq_len
        # Ensure max_start_index is not negative if len(img_paths) == min_seq_len
        if max_start_index < 0:
             max_start_index = 0

        bos = np.random.randint(0, max_start_index + 1) # +1 because randint is exclusive for the upper bound

        # Ensure eos doesn't exceed bounds and respects max_frames
        max_possible_len = min(self.max_frames, len(img_paths) - bos)
        if max_possible_len < min_seq_len: # Should not happen if max_start_index is correct, but double-check
             logger.warning(f"Sequence length calculation issue in {dir}. Retrying.")
             return self.__getitem__(_) # Retry if something went wrong

        # Sequence length will be between min_seq_len and max_possible_len
        seq_len = np.random.randint(min_seq_len, max_possible_len + 1)
        eos = bos + seq_len # eos is the exclusive end index for slicing frames

        selected_img_paths = img_paths[bos:eos]
        selected_raw_labels = raw_labels[bos:eos]

        # Load and process images, get dimensions from the first frame
        frames = []
        img_w, img_h = 0, 0
        for i, img_path in enumerate(selected_img_paths):
            try:
                img = Image.open(img_path).convert('RGB')
                if i == 0:
                    img_w, img_h = img.size
                    if img_w <= 0 or img_h <= 0:
                        logger.error(f"Invalid image dimensions ({img_w}, {img_h}) for {img_path}")
                        # Handle error: skip, retry, or raise
                        return self.__getitem__(_) # Retry sampling

                if self.transform:
                    img = self.transform(img) # Assume transform handles ToTensor and normalization
                else:
                    # Default transform to tensor and normalize
                    img = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0
                frames.append(img)
            except Exception as e:
                logger.error(f"Error loading image {img_path}: {e}. Retrying sample.")
                return self.__getitem__(_) # Retry sampling

        # Stack frames into tensor [T, C, H, W]
        frames = torch.stack(frames)
        T = frames.shape[0] # Actual sequence length T

        # Normalize labels using image dimensions
        normalized_labels = []
        for x, y, w, h in selected_raw_labels:
            # Handle potential division by zero if img_w or img_h were invalid
            norm_x = (x / img_w) if img_w > 0 else 0.0
            norm_y = (y / img_h) if img_h > 0 else 0.0
            norm_w = (w / img_w) if img_w > 0 else 0.0
            norm_h = (h / img_h) if img_h > 0 else 0.0
            # Clamp values to [0, 1] just in case of annotation errors
            normalized_labels.append(
                torch.tensor([norm_x, norm_y, norm_w, norm_h]).clamp(0.0, 1.0)
            )

        # Convert list of tensors to a single tensor [T, 4]
        target_bboxes = torch.stack(normalized_labels)

        # Final check for shapes
        if target_bboxes.shape[0] != T:
             logger.error(f"Shape mismatch: Frames T={T}, Target BBoxes shape={target_bboxes.shape}. Expected ({T}, 4)")
             return self.__getitem__(_) # Retry

        if target_bboxes.shape[1] != 4:
             logger.error(f"Shape mismatch: Target BBoxes shape={target_bboxes.shape}. Expected ({T}, 4)")
             return self.__getitem__(_) # Retry


        return {
            "frames": frames,           # Shape: [T, C, H, W]
            "target_bboxes": target_bboxes # Shape: [T, 4] (Normalized) - All boxes for the sequence
        }

if __name__ == "__main__":
    dataset = LaSotDataset("./datasets/lasot")
    sample = dataset[0]
    print("Frames: {} | Target BBoxes: {}".format(sample["frames"].shape,
                                                  sample["target_bboxes"].shape))
    # print(sample["target_bboxes"]) # Print the full sequence of boxes