import torch
from torch.utils.data import Dataset
import logging
import glob
import os
from random import choice
import cv2
import numpy as np
import json
import pycocotools.mask as mask_util

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def sav_decode_video(video_path: str) -> list[np.ndarray]:
    """
    Decode the video and return the RGB frames
    """
    video = cv2.VideoCapture(video_path)
    video_frames = []
    while video.isOpened():
        ret, frame = video.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            video_frames.append(frame)
        else:
            break
    return video_frames


class SyncAVDataset(Dataset):
    """
    Synchronous version of the SAV dataset loader.
    Loads video frames and annotations directly when requested.
    """

    def __init__(
        self,
        sav_dir: str,
        binary_task: bool = True,
        frames_per_sample: int = 32,
        annot_sample_rate: int = 4,
    ):
        self.sav_dir = sav_dir
        self.files = glob.glob(os.path.join(sav_dir, "*.mp4"))
        self.binary_task = binary_task
        self.annot_sample_rate = annot_sample_rate
        self.frames_per_sample = frames_per_sample

        if not self.files:
            logger.warning(f"No .mp4 files found in directory: {sav_dir}")
            raise ValueError("No frames;")

        logger.info(f"Found {len(self.files)} files in {sav_dir}")

    def _load_annotations(self, file_path: str) -> list[np.ndarray] | None:
        """Synchronously loads annotations for a given video file."""
        mask_path = file_path.replace(".mp4", "_auto.json")
        if not os.path.exists(mask_path):
            logger.warning(f"Annotation file not found: {mask_path}")
            return None  # Handle missing annotations gracefully

        try:
            with open(mask_path, "r") as f:
                annotations = json.load(f)
        except json.JSONDecodeError:
            logger.error(f"Error decoding JSON from {mask_path}")
            return None
        except Exception as e:
            logger.error(f"Error reading annotation file {mask_path}: {e}")
            return None

        # Check if annotations are empty or malformed
        if "masklet" not in annotations or not annotations["masklet"]:
            logger.warning(f"No 'masklet' data found or empty in {mask_path}")
            return None

        masks = []
        # Ensure there's at least one frame's annotation data
        if not annotations["masklet"][0]:
            logger.warning(f"Empty first frame annotation in {mask_path}")
            return None

        num_objects = len(annotations["masklet"][0])
        if num_objects == 0:
            logger.warning(f"No objects found in annotations for {mask_path}")
            return None

        # Choose object ID: random if binary task, else first object
        obj_id = choice(range(num_objects)) if self.binary_task else 0

        first_valid_mask_shape = None
        # Find the shape from the first available mask for the chosen object
        for frame_data in annotations["masklet"]:
            if obj_id < len(frame_data) and frame_data[obj_id] is not None:
                try:
                    first_valid_mask_shape = mask_util.decode(frame_data[obj_id]).shape[
                        :2
                    ]
                    break
                except Exception as e:
                    logger.error(
                        f"Error decoding mask for object {obj_id} in {mask_path}: {e}"
                    )
                    return None  # Cannot proceed without a valid mask shape

        if first_valid_mask_shape is None:
            logger.error(
                f"Could not find any valid mask for object {obj_id} in {mask_path} to determine shape."
            )
            return None

        h, w = first_valid_mask_shape

        for frame_data in annotations["masklet"]:
            # Handle cases where an object might disappear or annotation is missing
            if obj_id >= len(frame_data) or frame_data[obj_id] is None:
                # Use a zero mask with the determined shape
                masks.append(np.zeros((h, w), dtype=np.uint8))
                logger.debug(
                    f"Object {obj_id} missing in a frame for {mask_path}, using zero mask."
                )
            else:
                try:
                    decoded_mask = mask_util.decode(frame_data[obj_id])
                    # Ensure consistent shape, pad/crop if necessary (though ideally they match)
                    if decoded_mask.shape[:2] != (h, w):
                        logger.warning(
                            f"Inconsistent mask shape for object {obj_id} in {mask_path}. Expected {(h,w)}, got {decoded_mask.shape[:2]}. Using zero mask."
                        )
                        masks.append(np.zeros((h, w), dtype=np.uint8))
                    else:
                        masks.append(decoded_mask)
                except Exception as e:
                    logger.error(
                        f"Error decoding mask for object {obj_id} in frame for {mask_path}: {e}. Using zero mask."
                    )
                    masks.append(np.zeros((h, w), dtype=np.uint8))

        # Subsample masks based on annotation rate AFTER decoding all
        subsampled_masks = masks

        if not subsampled_masks:
            logger.warning(
                f"Subsampling resulted in empty mask list for {mask_path} with rate {self.annot_sample_rate}"
            )
            return None

        return subsampled_masks

    def __getitem__(self, index):
        """Synchronously loads and processes data for a given index."""
        if index >= len(self.files):
            raise IndexError("Index out of range")

        file_path = self.files[index]

        # Проверяем существование файла перед декодированием
        if not os.path.exists(file_path):
            logger.error(f"Video file not found: {file_path}")
            raise ValueError(f"Video file not found for index {index}: {file_path}")

        frames = sav_decode_video(file_path)
        # --- Validation ---
        if not frames:
            # Handle cases where video decoding failed (e.g., corrupted file)
            logger.error(f"Failed to decode video for index {index}: {file_path}")
            raise ValueError(f"Failed to load frames for index {index}")

        # Subsample frames *after* decoding all of them
        frames = frames[:: self.annot_sample_rate]
        annotations = self._load_annotations(file_path)

        assert len(frames) == len(
            annotations
        ), f"Number of frames and annotations do not match after subsampling. Frames: {len(frames)}, Annotations: {len(annotations)}"

        # Check if annotations loaded successfully before asserting length
        if annotations is None:
            # Handle cases where annotation loading failed or returned None
            logger.error(f"Failed to load annotations for index {index}: {file_path}")
            raise ValueError(f"Failed to load annotations for index {index}")

        # Assert length match after successful loading and subsampling
        if len(frames) != len(annotations):
            logger.error(
                f"Number of frames ({len(frames)}) and annotations ({len(annotations)}) "
                f"do not match after subsampling for index {index}: {file_path}"
            )
            raise ValueError(f"Frame and annotation count mismatch for index {index}")

        # Ensure enough frames/annotations for the sample
        num_available_frames = len(frames)
        num_available_annots = len(annotations)

        if (
            num_available_frames < self.frames_per_sample
            or num_available_annots < self.frames_per_sample
        ):
            logger.info(
                f"Skipping index {index}: Not enough frames/annotations after subsampling. "
                f"Got {num_available_frames} frames, {num_available_annots} annots. "
                f"Need at least {self.frames_per_sample} of each."
            )
            raise ValueError(
                f"Skipping index {index}: Not enough frames/annotations. Frames len: {num_available_frames}, Annots len: {num_available_annots}"
            )

        # --- Sampling ---
        # Select a starting point ensuring we have T frames and T masks
        max_start = (
            num_available_frames - self.frames_per_sample
        )  # Both are already subsampled

        if max_start < 0:
            raise ValueError(
                f"Cannot determine valid start for index {index} with {num_available_frames} frames/annotations for sample size {self.frames_per_sample}"
            )

        bos = np.random.randint(0, max_start + 1)  # Inclusive upper bound
        eos = bos + self.frames_per_sample

        # Select T frames and T masks from the *subsampled* lists
        selected_frames = frames[
            bos:eos
        ]  # Shape (T, H, W, C) - RGB from sav_decode_video
        selected_masks = annotations[bos:eos]  # Shape (T, H, W)

        # --- Data Transformation ---
        # Convert frames to tensor and permute: (T, H, W, C) -> (T, C, H, W)
        # Normalize frames to [0, 1]. Frames are already RGB.
        frames_np = np.stack(selected_frames)
        frames_tensor = (
            torch.tensor(frames_np, dtype=torch.float32).permute(0, 3, 1, 2) / 255.0
        )
        # BGR to RGB conversion is no longer needed here

        # Convert masks to tensor, ensure float, add channel dim: (T, H, W) -> (T, 1, H, W)
        # Normalize masks to [0, 1] if they are 0-255
        masks_np = np.stack(selected_masks)
        masks_tensor = torch.tensor(masks_np, dtype=torch.float32).unsqueeze(1)
        if masks_np.max() > 1:  # Check if normalization is needed
            masks_tensor = masks_tensor / 255.0
        # Optional: Ensure masks are binary (0 or 1)
        # masks_tensor = (masks_tensor > 0.5).float()

        # Split masks into past_masks (first T-1) and label (last one)
        # past_masks: (T-1, 1, H, W)
        past_masks_tensor = masks_tensor[:-1, :, :, :]
        # label: (1, 1, H, W)
        label_tensor = masks_tensor[-1:, :, :, :]  # Use slicing to keep dimension

        return {
            "frames": frames_tensor,  # Shape: (T, C, H, W)
            "masks": past_masks_tensor,  # Shape: (T-1, 1, H, W)
            "label": label_tensor,  # Shape: (1, 1, H, W)
        }

    def __len__(self):
        """Return the number of video files found."""
        return len(self.files)


# --- Example Usage ---
if __name__ == "__main__":
    # Create dummy data if it doesn't exist
    dummy_dir = "tfm/tests/sav_example_dummy_sync"  # Use a different dir name
    os.makedirs(dummy_dir, exist_ok=True)
    dummy_video_path = os.path.join(dummy_dir, "dummy_video_sync.mp4")
    dummy_annot_path = os.path.join(dummy_dir, "dummy_video_sync_auto.json")

    if not os.path.exists(dummy_video_path):
        print("Creating dummy video and annotations for sync dataset...")
        height, width = 64, 64
        # Increase frames_count slightly to ensure enough after subsampling if needed
        # Original video is 24fps, annotations are 6fps (sample_rate=4)
        # If we need 32 frames/annots at 6fps, we need 32 annotation points.
        # The video needs at least 32 * annot_sample_rate frames originally.
        # Let's use annot_sample_rate=4 for dummy data generation as in sav_utils example
        dummy_annot_sample_rate = 4
        frames_needed_for_sample = 32
        total_video_frames = (
            frames_needed_for_sample + 5
        ) * dummy_annot_sample_rate  # Add buffer
        total_annot_frames = total_video_frames // dummy_annot_sample_rate

        fps = 24  # Standard video fps
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(dummy_video_path, fourcc, fps, (width, height))
        print(f"Generating {total_video_frames} video frames...")
        for i in range(total_video_frames):
            frame = np.zeros((height, width, 3), dtype=np.uint8)
            # Adjust movement speed based on total frames
            x = min(max(0, int(i * width / total_video_frames)), width - 10)
            y = min(max(0, int(i * height / total_video_frames)), height - 10)
            cv2.rectangle(
                frame, (x, y), (x + 10, y + 10), (0, 255, 0), -1
            )  # Use BGR for writing
            writer.write(frame)
        writer.release()

        annotations = {"masklet": []}
        print(f"Generating {total_annot_frames} annotation frames...")
        # Generate annotations corresponding to the subsampled video frames
        for i in range(total_annot_frames):
            # Calculate the corresponding original video frame index
            original_frame_idx = i * dummy_annot_sample_rate
            mask = np.zeros((height, width), dtype=np.uint8)
            x = min(
                max(0, int(original_frame_idx * width / total_video_frames)), width - 10
            )
            y = min(
                max(0, int(original_frame_idx * height / total_video_frames)),
                height - 10,
            )
            cv2.rectangle(mask, (x, y), (x + 10, y + 10), 1, -1)  # Use 1 for mask value
            rle = mask_util.encode(np.asfortranarray(mask))
            rle["counts"] = rle["counts"].decode("utf-8")
            annotations["masklet"].append([rle])  # List containing one object's mask

        with open(dummy_annot_path, "w") as f:
            json.dump(annotations, f)
        print("Dummy data created.")

    # --- Test the SyncAVDataset ---
    print("\nTesting SyncAVDataset...")
    # Use annot_sample_rate=4 matching the dummy data generation and sav_utils default
    test_annot_sample_rate = 4
    test_frames_per_sample = 32
    sync_dataset = SyncAVDataset(
        dummy_dir,
        frames_per_sample=test_frames_per_sample,
        annot_sample_rate=test_annot_sample_rate,
    )
    print(f"Sync Dataset length: {len(sync_dataset)}")

    if len(sync_dataset) > 0:
        try:
            # Test getting the first item
            print("Getting item 0...")
            batch = sync_dataset[0]
            print(
                f"""Item 0:
Frames shape: {batch['frames'].shape}, dtype: {batch['frames'].dtype}, range: [{batch['frames'].min():.2f}, {batch['frames'].max():.2f}]
Masks shape: {batch['masks'].shape}, dtype: {batch['masks'].dtype}, range: [{batch['masks'].min():.2f}, {batch['masks'].max():.2f}]
Label shape: {batch['label'].shape}, dtype: {batch['label'].dtype}, range: [{batch['label'].min():.2f}, {batch['label'].max():.2f}]"""
            )

            # Verify shapes match expectations (T=32)
            # Frames are RGB (C=3)
            assert batch["frames"].shape == (test_frames_per_sample, 3, 64, 64)
            assert batch["masks"].shape == (
                test_frames_per_sample - 1,
                1,
                64,
                64,
            )  # T-1
            assert batch["label"].shape == (1, 1, 64, 64)
            print("Shapes are correct for item 0.")

            # Test getting another item if possible (only if multiple files exist)
            # if len(sync_dataset) > 1:
            #      print("\nGetting item 1...")
            #      batch1 = sync_dataset[1]
            #      print("Item 1 loaded successfully.")

        except ValueError as e:
            print(f"ValueError during testing: {e}")
            print("This might happen if dummy data is too short or loading failed.")
        except IndexError as e:
            print(f"IndexError during testing: {e}")
        except Exception as e:
            print(f"An unexpected error occurred during testing: {e}")
            import traceback

            traceback.print_exc()
    else:
        print("Sync Dataset is empty, cannot test __getitem__.")

    # Optional: Clean up dummy data
    # import shutil
    # shutil.rmtree(dummy_dir)
    # print("\nDummy sync data cleaned up.")
