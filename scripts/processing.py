#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import Image
import cv2
from cv_bridge import CvBridge

from gelsight import gsdevice
from gelsight import gs3drecon

width = rospy.get_param('~image_size/width')
height = rospy.get_param('~image_size/height')

nn_path = rospy.get_param('~nn_path')
nn_compute = rospy.get_param('~nn_compute', 'gpu')

nn = gs3drecon.Reconstruction3D(gs3drecon.Finger.R15)
nn.load_nn(nn_path, nn_compute)

vis3d = gs3drecon.Visualize3D(width, height, '', 0.0887)

bridge = CvBridge()

def image_cb(msg):
    frame = bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
    dm = nn.get_depthmap(frame, MASK_MARKERS_FLAG)
    vis3d.update(dm)

if __name__ == "__main__":
    rospy.init_node('processing')
    image_sub = rospy.Subscriber('image/rectified', Image, image_cb, queue_size=1)
    rospy.spin() 
