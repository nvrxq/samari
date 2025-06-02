#!/usr/bin/env python3

import sys
import cv2
import numpy as np
from PIL import Image
import vot_utils as vot
from vot_tracker import SamariVotTracker
import torch


def make_full_size(x, output_sz):
    '''
    zero-pad input x (right and down) to match output_sz
    x: numpy array e.g., binary mask
    output_sz: size of the output [width, height]
    '''
    if x.shape[0] == output_sz[1] and x.shape[1] == output_sz[0]:
        return x
    pad_x = output_sz[0] - x.shape[1]
    if pad_x < 0:
        x = x[:, :x.shape[1] + pad_x]
        # padding has to be set to zero, otherwise pad function fails
        pad_x = 0
    pad_y = output_sz[1] - x.shape[0]
    if pad_y < 0:
        x = x[:x.shape[0] + pad_y, :]
        # padding has to be set to zero, otherwise pad function fails
        pad_y = 0
    return np.pad(x, ((0, pad_y), (0, pad_x)), 'constant', constant_values=0)

def get_vot_mask(masks_list, image_width, image_height):
    id_ = 1
    masks_multi = np.zeros((image_height, image_width), dtype=np.float32)
    for mask in masks_list:
        m = make_full_size(mask, (image_width, image_height))
        masks_multi[m>0] = id_
        id_ += 1
    return masks_multi

class SamariVOT:
    def __init__(self, image, masks_list):
        """Initialize tracker with first frame and masks/regions
        Args:
            image: First frame
            masks_list: List of binary masks for initialization
        """
        self.tracker = SamariVotTracker(tracker_name="Samari-L")
        self.tracker.initialize(image, masks_list)


    def track(self, image):
        """Track objects in new frame
        Args:
            image: Current frame
        Returns:
            List of masks for each tracked object
        """
        # Convert image to PIL
        pil_image = Image.open(image)
        out_dict = self.tracker.track(pil_image)
        return out_dict["pred_mask"]


@torch.inference_mode()
def main():
    # Initialize VOT with mask support and multi-object tracking
    handle = vot.VOT("mask", multiobject=True)
    
    # Get initial objects (masks)
    objects = handle.objects()
    with open(f"/home/never/samari/VOT/mask.txt", "a") as f:
        f.write(str(len(objects)) + "\n")
    
    # Read first frame
    imagefile = handle.frame()
    if not imagefile:
        return
    image = Image.open(imagefile)
    init_masks = [make_full_size(m, (image.width, image.height)) for m in objects]
    with open(f"/home/never/samari/VOT/mask.txt", "a") as f:
        f.write(str(len(init_masks)) + "\n")
    # Initialize tracker
    tracker = SamariVOT(image, init_masks)
    while True:
        imagefile = handle.frame()
        if not imagefile:
            break
        # Get tracking result
        mask_list = tracker.track(imagefile)
        with open(f"/home/never/samari/VOT/mask.txt", "a") as f:
            f.write(str(len(mask_list)) + "\n")
        handle.report(mask_list)

if __name__ == "__main__":
    main() 