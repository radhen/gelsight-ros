#!/usr/bin/env python3

"""
Labels data for training a depth reconstruction model.
Can collect images using the 'record.py' script.

The dataset can only include the impression and rolling
of a single spherical object (like a marble). The first image
should also have no impression, for purposes of training from the
pixel-wise difference.

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

DEFAULT_RATE = 30
DEFAULT_FIELD_NAMES = ['R', 'G', 'B', 'x', 'y', 'gx', 'gy']
mpp = 0.08

if __name__ == "__main__":
    rospy.init_node("label")

    # Retrieve path where images are saved
    if not rospy.has_param("~input_path"):
        rospy.signal_shutdown("No input path provided. Please set input_path/.")
    input_path = rospy.get_param("~input_path")
    if input_path[-1] == "/":
        input_path = input_path[:len(input_path)-1]

    # Retrieve path where dataset will be saved    
    if not rospy.has_param("~output_path"):
        rospy.signal_shutdown("No output path provided. Please set output_path/.")
    output_path = rospy.get_param("~output_path")
    if output_path[-1] == "/":
        output_path = output_path[:len(output_path)-1]

    if not os.path.exists(output_path):
        rospy.logwarn("Output folder doesn't exist, will create it.")
        os.makedirs(output_path)
        
        if not os.path.exists(output_path):
            rospy.signal_shutdown(f"Failed to create output folder: {output_path}")
    output_file = output_path + "/gelsight-depth-dataset.csv"

    with open(output_file, 'w', newline='') as f:
        w = writer(f)
        w.writerow(DEFAULT_FIELD_NAMES)

    # Retrieve ball radius for building labels
    if not rospy.has_param("~ball_radius"):
        rospy.signal_shutdown("No ball radius provided. Please set ball_radius.")
    radius = rospy.get_param("~ball_radius")

    # Retrieve all images from input path
    imgs = [input_path + "/" + f for f in os.listdir(input_path) if os.path.isfile(os.path.join(input_path, f))]
    init_frame = cv2.imread(imgs[0])
    imgs = imgs[1:]                

    # Configure cv window
    cv2.namedWindow('label_data')


    # Main loop
    while not rospy.is_shutdown() and len(imgs) > 0:
        try:
            # Collect next frame and compute diff from initial frame
            current_frame = ((cv2.imread(imgs[0]) * 1.0) - init_frame) * 4 + 127
            current_frame[current_frame > 255] = 255
            current_frame[current_frame < 0] = 0
            current_frame = np.uint8(current_frame)

            # Convert to grayscale and find circles using hough transform
            grayscale_frame = cv2.cvtColor(current_frame, cv2.COLOR_RGB2GRAY)
            circles = cv2.HoughCircles(grayscale_frame, cv2.HOUGH_GRADIENT, 1, 20,
                param1=50,param2=30,minRadius=0,maxRadius=100)

            display_frame = current_frame.copy()
            if len(circles) > 0:
                # Draw first circle
                circle = np.uint16(np.around(circles))[0,:,0]
                cv2.circle(display_frame, (int(circle[0]), int(circle[1])),
                    int(circle[2]), (0,255,0), 2)
                cv2.circle(display_frame, (int(circle[0]), int(circle[1])),
                    2, (0, 0, 255), 3)

            cv2.imshow('label_data', display_frame) 
            
            k = cv2.waitKey(0)
            if k == ord('y'):
                # Compute labels from circle
                labels = []
                for x in range(current_frame.shape[0]):
                    for y in range(current_frame.shape[1]):

                        dx = (x - circle[0]) * mpp
                        dy = (y - circle[1]) * mpp

                        dist = radius**2 - dx**2 - dy**2
                        if dist <= 0.0 or dist > radius:
                            continue

                        r = current_frame[x, y, 0]
                        g = current_frame[x, y, 1]
                        b = current_frame[x, y, 2]

                        gx = (-dx) / math.sqrt(dist)
                        gy = (-dy) / math.sqrt(dist)

                        labels.append((r, g, b, x, y, gx, gy))

                # Write all labels to CSV file 
                with open(output_file, 'a', newline='') as f:
                    w = writer(f)
                    for label in labels:
                        w.writerow(label)
            elif k == ord('q'):
                break            

            # Move to next image
            imgs = imgs[1:]
        except rospy.ROSInterruptException:
            break

    cv2.destroyAllWindows()
    rospy.signal_shutdown(f"Data labeling completed, output to {output_file}")