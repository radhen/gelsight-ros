#!/usr/bin/env python3

import rospy
from std_msgs.msg import Header
from sensor_msgs import point_cloud2
from sensor_msgs.msg import Image, PointCloud2, PointField
import cv2
from cv_bridge import CvBridge
import numpy as np

from gelsight import gsdevice
from gelsight import gs3drecon

last_frame = None
bridge = CvBridge()

def image_cb(msg):
    global last_frame
    last_frame = bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')

def depth2pcl(px, py, mmpp, dm):
    points = np.zeros([px * py, 3])
    X, Y = np.meshgrid(np.arange(px) * mmpp, np.arange(py) * mmpp)
    points[:, 0] = np.ndarray.flatten(X)
    points[:, 1] = np.ndarray.flatten(Y)
    points[:, 2] = np.ndarray.flatten(dm)

    fields = [PointField('x', 0, PointField.FLOAT32, 1),
              PointField('y', 4, PointField.FLOAT32, 1),
              PointField('z', 8, PointField.FLOAT32, 1)]

    header = Header()
    header.frame_id = "gs"
    
    return point_cloud2.create_cloud(header, fields, points)


if __name__ == '__main__':
    rospy.init_node('gelsight_proc')

    use_http = rospy.get_param('~use_http', True)
    cam_url = rospy.get_param('~cam_url', '')

    if use_http and cam_url == '':
        rospy.logfatal('No "cam_url" provided when "use_http" is true')

    if not use_http: 
        image_sub = rospy.Subscriber('/image/raw', Image, image_cb, queue_size=1)
    else:
        dev = gsdevice.Camera(gsdevice.Finger.R15, cam_url)
        dev.connect()

    pcl_pub = rospy.Publisher('/pcl', PointCloud2, queue_size=1)

    width = rospy.get_param('~image_size/width')
    height = rospy.get_param('~image_size/height')

    nn_path = rospy.get_param('~nn_path')
    nn_compute = rospy.get_param('~nn_compute', 'gpu')

    nn = gs3drecon.Reconstruction3D(gs3drecon.Finger.R15)
    nn.load_nn(nn_path, nn_compute)

    #vis3d = gs3drecon.Visualize3D(120, 160, '', 0.0887)
    rate = rospy.Rate(30)
    while not rospy.is_shutdown():
        try:
            if use_http and dev.while_condition:
                last_frame = dev.get_image((60, 100, 375, 380))

            if last_frame is not None: 
                frame = last_frame.copy()
                dm = nn.get_depthmap(frame, False)
                pcl_pub.publish(depth2pcl(120, 160, 0.0887, dm))
                    
            rate.sleep()
        except rospy.ROSInterruptException:
            pass
