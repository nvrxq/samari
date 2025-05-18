#!/usr/bin/python
# coding: utf-8

import vot
import cv2
import numpy as np
from PIL import Image
import torch
import os

if __name__ == "__main__":
    gpu_use = "0"
    print("GPU use: {}".format(gpu_use))
    os.environ["CUDA_VISIBLE_DEVICES"] = "{}".format(gpu_use)

handle = vot.VOT("mask", multiobject=False)
objects = handle.objects()
imagefile = handle.frame()

image = cv2.imread(imagefile)
image_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

print("Image path: {} Objects: {}".format(imagefile, len(objects)))

from vot_tracker import SamariVotTracker

tracker = SamariVotTracker(tracker_name="sam21pp-L")

for obj_id, mask in enumerate(objects):
    if obj_id == 0:
        out_dict = tracker.initialize(image_pil, mask)
    else:
        print(
            f"Warning: Currently only supporting single object tracking. Skipping object {obj_id}"
        )
        break

while True:
    imagefile = handle.frame()
    if not imagefile:
        break

    image = cv2.imread(imagefile)
    image_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    out_dict = tracker.track(image_pil)

    mask_list = [out_dict["pred_mask"]]
    while len(mask_list) < len(objects):
        mask_list.append(np.zeros_like(mask_list[0]))

    handle.report(mask_list)
