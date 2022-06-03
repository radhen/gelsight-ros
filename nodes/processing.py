#!/usr/bin/env python3

import rospy
from std_msgs.messages import Bool
import pytouch
import pytouch.handlers import SensorHandler
from pytouch.sensors import GelsightSensor
from pytouch.tasks import TouchDetect
from sensor_msgs.msg import Image
import cv2
from cv_bridge import CvBridge

bridge = CvBridge()

last_frame = None
def image_cb(msg):
    last_frame = bridge.imgmsg_to_cv2(image_message, desired_encoding='passthrough')

sh = SensorHandler(0)
touch_detect = TouchDetect(GelsightSensor, zoo_model="touchdetect_resnet18")
touch_pub = rospy.Publisher('is_touching', Bool, queue_size=0)
image_sub = rospy.Subscriber('/test', Image, queue_size=1)

if __name__ == '__main__':
    try:
        rospy.init_node('gelsight_proc_node')
        rate = rospy.Rate(10)
        while not rospy.is_shutdown():
            if last_frame is not None:
                is_touching, certainty = touch_detect(last_frame)
            rate.sleep()
    except rospy.ROSInterruptException:
        pass
