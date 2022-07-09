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
import gelsight_ros as gsr
import math
from os import listdir
from os.path import isfile, join

DEFAULT_RATE = 30
DEFAULT_FIELD_NAMES = ['R', 'G', 'B', 'x', 'y', 'gx', 'gy']

mpp = 0.08
radius = None
circle = None
current_frame = None
output_file = None

def save_cb(state):
    global mpp, radius, circle, current_frame, output_file

    if not mpp:
        rospy.logfatal("Saved without meter-per-pixel. Exiting.")
    if not radius:
        rospy.logfatal("Saved without radius. Exiting.")
    if not circle:
        rospy.logfatal("Saved without circle. Exiting.")
    elif not current_frame:
        rospy.logfatal("Saved without current frame. Exiting.")
    elif not output_file:
        rospy.logfatal("Saved without output file. Exiting.")

    labels = []
    for x in range(current_frame.shape[0]):
        for y in range(current_frame.shape[1]):

            dx = (x - circle[0]) * mpp
            dy = (y - circle[1]) * mpp

            gx = (-dx) / math.sqrt(radius**2 - dx**2 - dy**2)
            gy = (-dy) / math.sqrt(radius**2 - dx**2 - dy**2)

            labels.append((current_frame[x, y, 0],
                           current_frame[x, y, 1],
                           current_frame[x, y, 2],
                           x, y, gx, gy))

    with open(output_file, 'a', newline='') as f:
        w = writer(f)
        for label in labels:
            w.writerow(label)
        w.close()

    current_frame = None

if __name__ == "__main__":
    rospy.init_node("label")
    rate = rospy.Rate(rospy.get_param("~rate", DEFAULT_RATE))

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

    # Retrieve ball radius for building labels
    if not rospy.has_param("~ball_radius"):
        rospy.signal_shutdown("No ball radius provided. Please set ball_radius.")
    radius = rospy.get_param("~ball_radius")

    # Retrieve all images from input path
    imgs = [f for f in listdir(input_path) if isfile(join(input_path, f))]
    init_frame = imgs[0]

    # Configure cv window
    cv2.namedWindow('label_data')
    cv2.createButton('Save', save_cb, None, cv2.QT_PUSH_BUTTON, 1)

    # Main loop
    while not rospy.is_shutdown() and len(imgs) > 0:
        try:
            if not current_frame:
                imgs = imgs[1:]                
                current_frame = cv2.imread(imgs[0], cv2.IMREAD_RGB)

            circles = cv.HoughCircles(current_frame, cv.HOUGH_GRADIENT, 1, 20,
                param1=50,param2=30,minRadius=0,maxRadius=0)

            display_frame = current_frame.copy()
            circle = np.uint16(np.around(circles))[0]
            cv.circle(display_frame, (circle[0], circle[1]), circle[2], (0,255,0), 2)
            cv.circle(display_frame, (circle[0], circle[1]), 2,(0,0,255), 3)

            cv2.imshow('label_data', display_frame) 
            cv2.waitKey(0)
            
            rate.sleep()
        except rospy.ROSInterruptException:
            break

    cv2.destroyAllWindows()