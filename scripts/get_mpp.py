#!/usr/bin/env python3

"""
Labels data for training a depth reconstruction model.
Can collect images using the 'record.py' script.

The dataset can only include the impression and rolling
of a single spherical object (like a marble). The first image
should also have no impression, for purposes of training from the
pixel-wise difference.

Directions:
- Press 'Y' to accept current circle label into the dataset
- Press 'N' to discard current image
- Press 'Q' to exit the program
- Click, drag and release to manually label a circle (replaces current label)

From a technical perspective, this script uses the known radius
of a spherical object to estimate the gradient at every contact
point. From the gradients, it can then generate a dataset in CSV
format which relates (R, G, B, x, y) -> (gx, gy). For inference,
you can then use poisson reconstruction to build the depth
map from gradients.

You can train a new model from the output dataset using the 'train.py' script.
"""

import csv
from csv import writer
import cv2
import gelsight_ros as gsr
import math
import numpy as np
import os
import rospy

dist = None
click_a = None

def click_cb(event, x, y, flags, param):
    global dist, click_a
    if event == cv2.EVENT_LBUTTONDOWN:
        if click_a is None:
            click_a = (x, y)
        else:
            px_dist = math.sqrt((x-click_a[0])**2 + (y-click_a[1])**2)
            print(f"MPP: {dist/px_dist}")
            exit()

if __name__ == "__main__":
    rospy.init_node("px_dist")

    dist = rospy.get_param("~dist")

    # Retrieve path where image is stored
    if not rospy.has_param("~input_path"):
        rospy.signal_shutdown("No input path provided. Please set input_path/.")
    input_path = rospy.get_param("~input_path")
    im = cv2.imread(input_path)

    # Configure cv window
    cv2.namedWindow('px_dist')
    cv2.setMouseCallback('px_dist', click_cb)

    cv2.imshow('px_dist', im) 
    cv2.waitKey(0)
    cv2.destroyAllWindows()