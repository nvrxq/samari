import argparse
from loguru import logger

import cv2
import gc
import numpy as np
import os
import sys
import torch
import yaml
from tqdm import tqdm
import json

sys.path.append("sam2")
from sam2.build_sam import build_sam2_video_predictor
from sam2.sam2_video_predictor import SAM2VideoPredictor

MASK_COLOR = [255, 0, 0]


def load_txt(gt_path):
    with open(gt_path, "r") as f:
        gt = f.readlines()
    boxes = {}
    for fid, line in enumerate(gt):
        x, y, w, h = map(float, line.split(","))
        x, y, w, h = int(x), int(y), int(w), int(h)
        boxes[fid] = ((x, y, x + w, y + h), 0)
    return boxes


def determine_model_cfg(model_path):
    if "large" in model_path:
        return "configs/sam2.1/sam2.1_hiera_l.yaml"
    elif "base_plus" in model_path:
        return "configs/sam2.1/sam2.1_hiera_b+.yaml"
    elif "small" in model_path:
        return "configs/sam2.1/sam2.1_hiera_s.yaml"
    elif "tiny" in model_path:
        return "configs/sam2.1/sam2.1_hiera_t.yaml"
    else:
        raise ValueError("Please specify the model name")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--filter-config", "-f", type=str, default="filter.yaml")
    parser.add_argument(
        "--video-path", "-v", type=str, default="assets/1917-for-sam2.mp4"
    )
    parser.add_argument("--output-path", "-o", type=str, default="mcmc_demo_result.mp4")
    parser.add_argument(
        "--label-file", "-l", type=str, default="assets/1917-for-sam2.label"
    )
    parser.add_argument(
        "--model-path", "-m", type=str, default="sam2/models/sam2.1_hiera_b+.pth"
    )
    parser.add_argument(
        "--debug-output-path", "-d", type=str, default="debug_output.json"
    )
    return parser.parse_args()


def main(args):
    sam2_cfg = determine_model_cfg(args.model_path)
    predictor = build_sam2_video_predictor(sam2_cfg, args.model_path, device="cuda:0")
    with open(args.filter_config, "r") as f:
        filter_config = yaml.safe_load(f)
    predictor.filter_from_config(filter_config)

    boxes = load_txt(args.label_file)

    video_path = args.video_path
    box = load_txt(args.label_file)
    fps = 30
    cap = cv2.VideoCapture(args.video_path)
    if not cap.isOpened():
        logger.error("Could not open video file.")
    else:
        logger.info("Video file opened successfully!")
    fps = cap.get(cv2.CAP_PROP_FPS)
    loaded_frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            logger.error(ret)
            break
        loaded_frames.append(frame)
    cap.release()
    height, width = loaded_frames[0].shape[:2]

    if len(loaded_frames) == 0:
        raise ValueError("No frames were loaded from the video.")

    logger.info(f"Loaded {len(loaded_frames)} frames from the video.")
    logger.info(f"Video resolution: {width}x{height}")
    logger.info(f"FPS: {fps}")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(args.output_path, fourcc, fps, (width, height))
    saved_for_video = []
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
        state = predictor.init_state(video_path, offload_video_to_cpu=True)
        bbox, _ = boxes[0]
        _, _, masks = predictor.add_new_points_or_box(
            state, box=bbox, frame_idx=0, obj_id=0
        )

        for frame_idx, object_ids, masks in tqdm(predictor.propagate_in_video(state)):
            mask_to_vis = {}
            bbox_to_vis = {}

            for obj_id, mask in zip(object_ids, masks):
                mask = mask[0].cpu().numpy()
                mask = mask > 0.0
                non_zero_indices = np.argwhere(mask)
                if len(non_zero_indices) == 0:
                    bbox = [0, 0, 0, 0]
                else:
                    y_min, x_min = non_zero_indices.min(axis=0).tolist()
                    y_max, x_max = non_zero_indices.max(axis=0).tolist()
                    bbox = [x_min, y_min, x_max - x_min, y_max - y_min]
                bbox_to_vis[obj_id] = bbox
                mask_to_vis[obj_id] = mask

            img = loaded_frames[frame_idx]
            for obj_id, mask in mask_to_vis.items():
                mask_img = np.zeros((height, width, 3), np.uint8)
                mask_img[mask] = MASK_COLOR[(obj_id + 1) % len(MASK_COLOR)]
                img = cv2.addWeighted(img, 1, mask_img, 0.2, 0)

            for obj_id, bbox in bbox_to_vis.items():
                cv2.rectangle(
                    img,
                    (bbox[0], bbox[1]),
                    (bbox[0] + bbox[2], bbox[1] + bbox[3]),
                    MASK_COLOR[obj_id % len(MASK_COLOR)],
                    2,
                )
            out.write(img)
            saved_for_video.append(img.copy())
        out.release()

    gc.collect()
    torch.clear_autocast_cache()
    torch.cuda.empty_cache()
    # Dump debug dict from filter.
    with open(args.debug_output_path, "w") as f:
        json.dump(predictor.mcmc.history, f)


if __name__ == "__main__":
    args = parse_args()
    main(args)
