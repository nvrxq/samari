# Credit: https://github.com/jovanavidenovic/DAM4SAM/blob/master/dam4sam_tracker.py
import numpy as np
import yaml
import torch
import torchvision.transforms.functional as F

from vot.region.raster import calculate_overlaps
from vot.region.shapes import Mask
from vot.region import RegionType
from sam2.build_sam import build_sam2_video_predictor
from collections import OrderedDict
import random
import os

from pathlib import Path

config_path = Path(__file__).parent / "dam4sam_config.yaml"
with open(config_path) as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

seed = config["seed"]
random.seed(seed)
os.environ["PYTHONHASHSEED"] = str(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)


def get_cfg(tracker_name):
    if tracker_name == "Samari-B":
        return ("configs/sam2.1/sam2.1_hiera_b+.yaml", "sam2.1_hiera_b+_samari.pt")
    elif tracker_name == "Samari-L":
        return ("configs/sam2.1/sam2.1_hiera_l.yaml", "sam2.1_hiera_l_samari.pt")
    elif tracker_name == "Samari-S":
        return ("configs/sam2.1/sam2.1_hiera_s.yaml", "sam2.1_hiera_s_samari.pt")
    else:
        raise ValueError(f"Invalid tracker name: {tracker_name}")


class SamariVotTracker:
    def __init__(self, tracker_name="Samari-B"):
        """
        Constructor for the SamariVotTracker.

        Args:
        - tracker_name (str): Name of the tracker to use
        """
        self.checkpoint, self.model_cfg = get_cfg(tracker_name)

        # Image preprocessing parameters
        self.input_image_size = 1024
        self.img_mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32)[
            :, None, None
        ]
        self.img_std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32)[
            :, None, None
        ]

        # Initialize SAM2 predictor with filter
        self.predictor = build_sam2_video_predictor(
            self.model_cfg, self.checkpoint, device="cuda:0"
        )
        self.tracking_times = []

    def filter_from_config(self, config_path):
        """
        Configure the filter from a YAML file.

        Args:
        - config_path (str): Path to the filter configuration YAML file
        """
        with open(config_path, "r") as f:
            filter_config = yaml.safe_load(f)
        self.predictor.filter_from_config(filter_config)

    def _prepare_image(self, img_pil):
        """Prepare image for input to SAM2."""
        img = torch.from_numpy(np.array(img_pil)).to(self.inference_state["device"])
        img = img.permute(2, 0, 1).float() / 255.0
        img = F.resize(img, (self.input_image_size, self.input_image_size))
        img = (img - self.img_mean) / self.img_std
        return img

    @torch.inference_mode()
    def init_state_tw(self):
        """Initialize inference state."""
        compute_device = torch.device("cuda")
        inference_state = {}
        inference_state["images"] = None
        inference_state["num_frames"] = 0
        inference_state["video_height"] = None
        inference_state["video_width"] = None
        inference_state["device"] = compute_device
        inference_state["storage_device"] = compute_device
        inference_state["output_dict"] = {
            "cond_frame_outputs": {},
            "non_cond_frame_outputs": {},
        }

        self.img_mean = self.img_mean.to(compute_device)
        self.img_std = self.img_std.to(compute_device)

        return inference_state

    @torch.inference_mode()
    def initialize(self, image, init_mask, bbox=None):
        """
        Initialize the tracker with the first frame and mask.

        Args:
        - image (PIL Image): First frame of the video.
        - init_mask (numpy array): Binary mask for initialization
        - bbox (list): Optional bounding box for initialization

        Returns:
        - out_dict (dict): Dictionary containing the mask for initialization frame.
        """
        if type(init_mask) is list:
            init_mask = init_mask[0]
        self.frame_index = 0

        self.img_width = image.width
        self.img_height = image.height
        self.inference_state = self.init_state_tw()
        video_width, video_height = image.size
        self.inference_state["video_height"] = video_height
        self.inference_state["video_width"] = video_width
        prepared_img = self._prepare_image(image)
        self.inference_state["images"] = {0: prepared_img}
        self.inference_state["num_frames"] = 1
        self.predictor.reset_state(self.inference_state)

        if init_mask is None and bbox is not None:
            init_mask = self.estimate_mask_from_box(bbox)
        elif init_mask is None and bbox is None:
            raise ValueError("Error: initialization state (bbox or mask) is not given.")

        _, _, out_mask_logits = self.predictor.add_new_mask(
            inference_state=self.inference_state,
            frame_idx=0,
            obj_id=0,
            mask=init_mask,
        )

        m = (out_mask_logits[0, 0] > 0).float().cpu().numpy().astype(np.uint8)
        self.inference_state["images"].pop(self.frame_index)

        return {"pred_mask": m}

    @torch.inference_mode()
    def track(self, image, init=False):
        """
        Track object in the next frame.

        Args:
        - image (PIL Image): Next frame of the video.
        - init (bool): Whether current frame is initialization frame.

        Returns:
        - out_dict (dict): Dictionary containing predicted mask.
        """
        torch.cuda.empty_cache()
        prepared_img = self._prepare_image(image).unsqueeze(0)
        if not init:
            self.frame_index += 1
            self.inference_state["num_frames"] += 1
        self.inference_state["images"][self.frame_index] = prepared_img

        for out in self.predictor.propagate_in_video(
            self.inference_state,
            start_frame_idx=self.frame_index,
            max_frame_num_to_track=0,
            return_all_masks=True,
        ):
            if len(out) == 3:
                _, _, out_mask_logits = out
            else:
                _, _, out_mask_logits, _ = out

            m = (out_mask_logits[0][0] > 0.0).float().cpu().numpy().astype(np.uint8)
            self.inference_state["images"].pop(self.frame_index)
            return {"pred_mask": m}

    def estimate_mask_from_box(self, bbox):
        """
        Estimate mask from bounding box using SAM2.

        Args:
        - bbox (list): Bounding box coordinates [x, y, w, h]

        Returns:
        - init_mask (numpy array): Estimated binary mask
        """
        (
            _,
            _,
            current_vision_feats,
            current_vision_pos_embeds,
            feat_sizes,
        ) = self.predictor._get_image_feature(self.inference_state, 0, 1)

        box = np.array([bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]])[
            None, :
        ]
        box = torch.as_tensor(
            box, dtype=torch.float, device=current_vision_feats[0].device
        )

        from sam2.utils.transforms import SAM2Transforms

        _transforms = SAM2Transforms(
            resolution=self.predictor.image_size,
            mask_threshold=0.0,
            max_hole_area=0.0,
            max_sprinkle_area=0.0,
        )
        unnorm_box = _transforms.transform_boxes(
            box, normalize=True, orig_hw=(self.img_height, self.img_width)
        )

        box_coords = unnorm_box.reshape(-1, 2, 2)
        box_labels = torch.tensor([[2, 3]], dtype=torch.int, device=unnorm_box.device)
        box_labels = box_labels.repeat(unnorm_box.size(0), 1)
        concat_points = (box_coords, box_labels)

        sparse_embeddings, dense_embeddings = self.predictor.sam_prompt_encoder(
            points=concat_points, boxes=None, masks=None
        )

        high_res_features = []
        for i in range(2):
            _, b_, c_ = current_vision_feats[i].shape
            high_res_features.append(
                current_vision_feats[i]
                .permute(1, 2, 0)
                .view(b_, c_, feat_sizes[i][0], feat_sizes[i][1])
            )

        img_embed = current_vision_feats[2]
        if self.predictor.directly_add_no_mem_embed:
            img_embed = img_embed + self.predictor.no_mem_embed
        _, b_, c_ = img_embed.shape
        img_embed = img_embed.permute(1, 2, 0).view(
            b_, c_, feat_sizes[2][0], feat_sizes[2][1]
        )

        low_res_masks, iou_predictions, _, _ = self.predictor.sam_mask_decoder(
            image_embeddings=img_embed,
            image_pe=self.predictor.sam_prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False,
            repeat_image=False,
            high_res_features=high_res_features,
        )

        masks = _transforms.postprocess_masks(
            low_res_masks, (self.img_height, self.img_width)
        )
        masks = masks > 0
        return masks.squeeze(0).float().detach().cpu().numpy()[0]
