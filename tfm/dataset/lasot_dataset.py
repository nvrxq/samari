from torch.utils.data import Dataset
import torch
import glob
import os
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as patches
from copy import deepcopy

import albumentations as A
from PIL import Image
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LaSotDataset(Dataset):
    def __init__(
        self,
        root_dir,
        transform=None,
        steps_per_epoch: int = 100,
        max_frames: int = 60,
        mode: str = None,
    ):
        if mode is None:
            raise ValueError("Mode must be either 'prediction' or 'correction'")
        self.mode = mode
        self.root_dir = root_dir
        self.transform = transform
        self.files = glob.glob(os.path.join(root_dir, "*"))
        self.dir2labels = {}
        self.max_frames = max_frames
        self.steps_per_epoch = steps_per_epoch
        self._create_labels()
        logger.info(
            f"Your dataset setup:\nRoot dir: {self.root_dir}\nTransform: {self.transform}\nSteps per epoch: {self.steps_per_epoch}\nMax frames: {self.max_frames}\nMode: {self.mode}"
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

    def __len__(self):
        return self.steps_per_epoch

    def _shift_box(self, box: tuple, height: int, width: int) -> tuple:
        """
        Perturbs a bounding box using Albumentations library to create training examples for correction.
        
        Args:
            box: Original bounding box in (x, y, w, h) format
            height: Image height
            width: Image width
            
        Returns:
            Perturbed bounding box in (x, y, w, h) format
        """
        x, y, w, h = box
        
        # Convert from (x, y, w, h) to (x_min, y_min, x_max, y_max) format for Albumentations
        x_min, y_min = x, y
        x_max, y_max = x + w, y + h
        
        # Create a dummy image matching the original dimensions
        dummy_image = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Define a light augmentation pipeline for gentle perturbations
        # Using very small values to ensure subtle changes
        aug = A.Compose([
            A.OneOf([
                # Subtle geometric transformations
                A.ShiftScaleRotate(
                    shift_limit=0.5,  # Very small shift
                    scale_limit=0.5,  # Very small scale
                    rotate_limit=0.7,    # No rotation
                    p=0.7
                ),
                # Small perspective changes
                A.Perspective(scale=(0.1, 0.1), p=0.3),
            ], p=0.8),
            
            # Random cropping that will slightly adjust the box
            A.RandomSizedBBoxSafeCrop(
                height=height,
                width=width,
                erosion_rate=0.1,
                p=0.2
            ),
            
            # Occasionally adjust the box proportions slightly
            A.OneOf([
                A.HorizontalFlip(p=0.0),  # No flipping, just for the bounding box adjustments
                A.VerticalFlip(p=0.0),    # No flipping, just for the bounding box adjustments
                A.RandomSizedBBoxSafeCrop(
                    height=height,
                    width=width,
                    erosion_rate=0.1,
                    p=1.0
                ),
            ], p=0.2),
        ], 
        bbox_params=A.BboxParams(
            format='pascal_voc',  # xmin, ymin, xmax, ymax
            label_fields=['labels']
        ))
        
        # Apply the augmentation to get the perturbed box
        try:
            augmented = aug(
                image=dummy_image,
                bboxes=[(x_min, y_min, x_max, y_max)],
                labels=['bbox']
            )
            
            # Extract the transformed bounding box
            if augmented['bboxes']:
                aug_x_min, aug_y_min, aug_x_max, aug_y_max = augmented['bboxes'][0]
                
                # Convert back to (x, y, w, h) format
                aug_w = aug_x_max - aug_x_min
                aug_h = aug_y_max - aug_y_min
                
                # If the box barely changed, add a tiny random perturbation
                if abs(aug_x_min - x_min) < 1 and abs(aug_y_min - y_min) < 1 and abs(aug_w - w) < 1 and abs(aug_h - h) < 1:
                    aug_x_min += np.random.uniform(-2, 2)
                    aug_y_min += np.random.uniform(-2, 2)
                    aug_w += np.random.uniform(-1, 1) 
                    aug_h += np.random.uniform(-1, 1)
                
                return (aug_x_min, aug_y_min, aug_w, aug_h)
        except Exception as e:
            # If any error occurs, fall back to a simple random perturbation
            x += np.random.uniform(-3, 3)
            y += np.random.uniform(-3, 3)
            w += np.random.uniform(-1.5, 1.5)
            h += np.random.uniform(-1.5, 1.5)
        
        # Ensure minimum box size and keep box within image boundaries
        w = max(10, min(width - x, w))
        h = max(10, min(height - y, h))
        x = max(0, min(width - w, x))
        y = max(0, min(height - h, y))
        
        return (x, y, w, h)

    def _transform_box_and_image(self, img, label_xywh, img_path, img_w, img_h, allow_full_box=False):
        """
        Transforms both image and bounding box using the dataset's transform.
        
        Args:
            img: PIL Image to transform
            label_xywh: Bounding box in (x, y, w, h) format
            img_path: Path to image (for error messaging)
            img_w: Original image width
            img_h: Original image height
            
        Returns:
            tuple: (transformed_image_tensor, transformed_box_xywh)
        """
        x, y, w, h = label_xywh
        x_min, y_min = x, y
        x_max, y_max = x + w, y + h

        # Ensure the box is within image boundaries
        x_min = max(0, x_min)
        y_min = max(0, y_min)
        x_max = min(img_w, x_max)
        y_max = min(img_h, y_max)
        if not allow_full_box:
            if x_min >= x_max or y_min >= y_max:
                logger.warning(f"Invalid bbox [{x_min}, {y_min}, {x_max}, {y_max}] for {img_path}")
                raise ValueError(f"Invalid bbox [{x_min}, {y_min}, {x_max}, {y_max}] for {img_path}")

        transformed = self.transform(
            image=np.array(img),
            bboxes=[[x_min, y_min, x_max, y_max]],
            labels=[0],
        )
        
        img_tensor = transformed["image"]
        transformed_bboxes_pascal = transformed["bboxes"]

        if not transformed_bboxes_pascal:
            raise ValueError(f"Bounding box disappeared after transform for {img_path}")
            
        t_x_min, t_y_min, t_x_max, t_y_max = transformed_bboxes_pascal[0]
        t_w = t_x_max - t_x_min
        t_h = t_y_max - t_y_min
        transformed_box_xywh = (t_x_min, t_y_min, t_w, t_h)
        
        return img_tensor, transformed_box_xywh

    def __getitem__(self, _):
        dir_path = np.random.choice(self.files)
        label_info = self.dir2labels[dir_path]
        img_paths = label_info["imgs"]
        raw_labels = self.load_labels(label_info["label_file"])

        if len(img_paths) != len(raw_labels):
            logger.warning(
                f"Skipping {dir_path}: mismatched images ({len(img_paths)}) and labels ({len(raw_labels)})"
            )
            return self._return_dummy_item()

        min_seq_len = 2
        if len(img_paths) < min_seq_len:
            logger.warning(
                f"Skipping {dir_path}: too few frames ({len(img_paths)} < {min_seq_len})"
            )
            return self._return_dummy_item()

        max_start_index = len(img_paths) - min_seq_len
        if max_start_index < 0:
            logger.warning(
                f"Skipping {dir_path}: cannot select min sequence length ({len(img_paths)} < {min_seq_len})"
            )
            return self._return_dummy_item()
        """ Select sequence of frames. Frames must be sorted by time."""
        bos = np.random.randint(0, max_start_index + 1)
        max_possible_len = min(self.max_frames, len(img_paths) - bos)
        seq_len = np.random.randint(min_seq_len, max_possible_len + 1)
        eos = bos + seq_len

        selected_imgs = img_paths[bos:eos]
        selected_labels_xywh = raw_labels[bos:eos]
        if self.mode == "correction":
            box_to_augment = selected_labels_xywh[-1]
            target_box = deepcopy(selected_labels_xywh[-1])

            dummy_image = Image.open(selected_imgs[-1])
            dummy_image_h, dummy_image_w = dummy_image.size
            box_to_augment = self._shift_box(
                box_to_augment, dummy_image_h, dummy_image_w
            )
            selected_labels_xywh[-1] = box_to_augment
            selected_labels_xywh.append(target_box)
        # ---------------------------------#
        frames = []
        transformed_labels_xywh = []
        img_w, img_h = None, None
        for idx, (img_path, label_xywh) in enumerate(
            zip(selected_imgs, selected_labels_xywh)
        ):
            try:
                img = Image.open(img_path).convert("RGB")
                if idx == 0:
                    img_w, img_h = img.size
                    if img_w <= 0 or img_h <= 0:
                        logger.error(
                            f"Invalid original image dimensions {img.size} for {img_path}"
                        )
                        raise ValueError(
                            "Invalid original image dimensions {img.size} for {img_path}"
                        )

                if self.transform:
                    img_tensor, transformed_box = self._transform_box_and_image(
                        img, label_xywh, img_path, img_w, img_h, allow_full_box=idx == len(selected_imgs) - 1
                    )
                    frames.append(img_tensor)
                    transformed_labels_xywh.append(transformed_box)
                else:
                    img_tensor = torch.from_numpy(np.array(img)).permute(2, 0, 1).to(torch.float16) / 255.0
                    frames.append(img_tensor)
                    transformed_labels_xywh.append(label_xywh)
            except Exception as e:
                logger.error(f"Error processing {img_path}: {e}")
                raise e
        _, target_box = self._transform_box_and_image(
            Image.open(selected_imgs[-1]), selected_labels_xywh[-1], selected_imgs[-1], img_w, img_h
        )


        transformed_labels_xywh.append(target_box) # Add target box to labels to normalize.
        if not frames:
            logger.warning(f"No valid frames found in {dir_path}")
            raise ValueError("No valid frames found in {dir_path}")

        normalized_cxcywh_labels = []
        final_img_h, final_img_w = frames[0].shape[1], frames[0].shape[2]

        if final_img_w <= 0 or final_img_h <= 0:
            logger.error(
                f"Invalid final image dimensions ({final_img_w}, {final_img_h}) for {dir_path}"
            )
            raise ValueError(
                "Invalid final image dimensions ({final_img_w}, {final_img_h}) for {dir_path}"
            )

        for x, y, w, h in transformed_labels_xywh:
            norm_x = x / final_img_w
            norm_y = y / final_img_h
            norm_w = w / final_img_w
            norm_h = h / final_img_h

            norm_cx = norm_x + norm_w / 2
            norm_cy = norm_y + norm_h / 2

            bbox_tensor = torch.tensor(
                [norm_cx, norm_cy, norm_w, norm_h], dtype=torch.float16
            )
            bbox_tensor[0:2] = torch.clamp(bbox_tensor[0:2], 0.0, 1.0)
            bbox_tensor[2:4] = torch.clamp(bbox_tensor[2:4], 0.0, 1.0)

            normalized_cxcywh_labels.append(bbox_tensor)
        if len(frames) + 1 != len(normalized_cxcywh_labels): # +1 for target box.
            logger.error(
                f"Frame count ({len(frames)}) mismatch with label count ({len(normalized_cxcywh_labels)}) for {dir_path}"
            )
            raise ValueError(
                "Frame count ({len(frames)}) mismatch with label count ({len(normalized_cxcywh_labels)}) for {dir_path}"
            )

        frames_tensor = torch.stack(frames)
        target_bboxes_tensor = torch.stack(normalized_cxcywh_labels)
        return {
            "frames": frames_tensor,
            "past": target_bboxes_tensor[:-1],
            "labels": target_bboxes_tensor[-1],
        }


def visualize_sample(sample, show_labels=True):
    """
    Visualizes a sample from the dataset showing frames with bounding boxes.

    Args:
        sample: Dictionary containing 'frames' and 'target_bboxes'
               (and optionally 'labels' for perturbed boxes)
        show_labels: Whether to show the labels (perturbed boxes) if available
    """
    frames = sample["frames"]
    target_bboxes = sample["past"]

    for t in range(len(frames)):
        frame = frames[t].permute(1, 2, 0).numpy()
        # Convert from normalized cx,cy,w,h to pixel x,y,w,h
        img_h, img_w = frame.shape[0], frame.shape[1]

        # Create figure with 1 or 2 subplots depending on if we have labels
        if t == len(frames) - 1:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
            axes = [ax1, ax2]
            titles = ["Target Bounding Box", "Perturbed Bounding Box"]
        else:
            fig, ax1 = plt.subplots(1, figsize=(6, 6))
            axes = [ax1]
            titles = ["Target Bounding Box"]

        # Show original target bounding box
        ax1.imshow(frame)
        bbox = target_bboxes[t].cpu().numpy()
        # Convert from normalized cx,cy,w,h to pixel x,y,w,h
        x = (bbox[0] - bbox[2] / 2) * img_w
        y = (bbox[1] - bbox[3] / 2) * img_h
        w = bbox[2] * img_w
        h = bbox[3] * img_h
        print("X", x, "Y", y, "W", w, "H", h)

        rect = patches.Rectangle(
            (x, y), w, h, linewidth=2, edgecolor="r", facecolor="none"
        )
        ax1.add_patch(rect)
        ax1.set_title(f"{titles[0]} - Frame {t + 1}")
        ax1.axis("off")

        # Show perturbed bounding box if available
        if t == len(frames) - 1:
            ax2.imshow(frame)
            label_bbox = sample["labels"].cpu().numpy()
            # Convert from normalized cx,cy,w,h to pixel x,y,w,h
            x = (label_bbox[0] - label_bbox[2] / 2) * img_w
            y = (label_bbox[1] - label_bbox[3] / 2) * img_h
            w = label_bbox[2] * img_w
            h = label_bbox[3] * img_h
            print("Label X", x, "Label Y", y, "Label W", w, "Label H", h)
            rect = patches.Rectangle(
                (x, y), w, h, linewidth=2, edgecolor="g", facecolor="none"
            )
            ax2.add_patch(rect)
            ax2.set_title(f"{titles[1]} - Frame {t + 1}")
            ax2.axis("off")

        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    # Example usage with necessary transforms for testing
    img_size = [256, 256]  # Example size, match your config
    img_mean = [0.485, 0.456, 0.406]
    img_std = [0.229, 0.224, 0.225]

    test_transform = A.Compose(
        [
            A.Resize(height=img_size[0], width=img_size[1]),
            A.ToTensorV2(),  # Converts image to PyTorch tensor (C, H, W)
        ],
        bbox_params=A.BboxParams(
            format="pascal_voc",  # Input format expected by Compose
            min_visibility=0.1,
            label_fields=["labels"],  # Use 'labels' as the field name
        ),
    )

    # Provide the path to your LaSOT dataset directory
    dataset_root = "./datasets/lasot"  # CHANGE THIS TO YOUR PATH
    if not os.path.exists(dataset_root):
        print(f"Dataset directory not found: {dataset_root}")
        print("Please update the 'dataset_root' variable in the script.")
    else:
        print(f"Loading dataset from: {dataset_root}")
        dataset = LaSotDataset(
            dataset_root,
            transform=test_transform,
            steps_per_epoch=10,
            max_frames=3,
            mode="correction",
        )

        # Test getting a sample
        try:
            print("Attempting to get a sample...")
            sample = dataset[0]
            print("Sample retrieved successfully.")
            # Verify shapes and format
            frames = sample["frames"]
            bboxes = sample["past"]
            print("Past", bboxes)
            print("Labels", sample["labels"])
            print(f"Frames shape: {frames.shape}")  # Should be (T, C, H, W)
            print(f"BBoxes shape: {bboxes.shape}")  # Should be (T, 4)
            print(
                f"BBoxes format (first frame): {bboxes[0]}"
            )  # Should be cxcywh, normalized [0,1]
            visualize_sample(sample)
        except Exception as e:
            print(f"Error getting or processing sample: {e}")
            import traceback

            traceback.print_exc()
