#!/usr/bin/env python3

import sys
import cv2
import numpy as np
from PIL import Image
import vot_utils as vot
from vot_tracker import SamariVotTracker

class SamariVOT:
    def __init__(self, image, masks_list):
        """Initialize tracker with first frame and masks/regions
        Args:
            image: First frame
            masks_list: List of binary masks for initialization
        """
        self.tracker = SamariVotTracker(tracker_name="Samari-L")
        
        # Initialize for each object
        self.trackers = []
        for mask in masks_list:
            # Convert image to PIL
            pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            
            # Initialize tracker with mask
            out_dict = self.tracker.initialize(pil_image, mask)
            self.trackers.append({
                'tracker': self.tracker,
                'mask': mask
            })

    def track(self, image):
        """Track objects in new frame
        Args:
            image: Current frame
        Returns:
            List of masks for each tracked object
        """
        # Convert image to PIL
        pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        
        mask_list = []
        for tracker_info in self.trackers:
            # Track each object
            out_dict = tracker_info['tracker'].track(pil_image)
            mask_list.append(out_dict["pred_mask"])
            
        return mask_list

def main():
    # Initialize VOT with mask support and multi-object tracking
    handle = vot.VOT("mask", multiobject=True)
    
    # Get initial objects (masks)
    objects = handle.objects()
    
    # Read first frame
    imagefile = handle.frame()
    if not imagefile:
        return
    image = cv2.imread(imagefile)
    print('Image path: {} Objects: {}'.format(imagefile, len(objects)))
    
    # Initialize tracker
    tracker = SamariVOT(image, objects)

    # Main tracking loop
    while True:
        imagefile = handle.frame()
        if not imagefile:
            break
        
        # Read next frame
        image = cv2.imread(imagefile)
        
        # Get tracking result
        mask_list = tracker.track(image)
        
        # Report back to VOT
        handle.report(mask_list)

if __name__ == "__main__":
    main() 