#!/usr/bin/env python3

import rospy
import math
from std_msgs.msg import Header, Float32
from sensor_msgs import point_cloud2
from sensor_msgs.msg import Image, PointCloud2, PointField
from geometry_msgs.msg import PoseStamped
import cv2
from cv_bridge import CvBridge
import numpy as np
from numpy import linalg as LA

from gelsight import gsdevice
from gelsight import gs3drecon

last_frame = None
bridge = CvBridge()

def image_cb(msg):
    global last_frame
    last_frame = bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')

def depth2pcl(px, py, mmpp, dm):
    points = []
    for i in range(px):
        for j in range(py):
            points.append((i*mmpp/100.0, j*mmpp/100.0, dm[j, i]/1000.0, int(0)))


    fields = [PointField('x', 0, PointField.FLOAT32, 1),
              PointField('y', 4, PointField.FLOAT32, 1),
              PointField('z', 8, PointField.FLOAT32, 1),
              PointField('rgb', 12, PointField.UINT32, 1)]

    header = Header()
    header.frame_id = "map"
    
    return point_cloud2.create_cloud(header, fields, points)

def pca(pcl):
    pcl = pcl.reshape(-1, 2).astype(np.float64)
    mv = np.mean(pcl, 0).reshape(2, 1)
    pcl -= mv.T
    w, v = LA.eig(np.dot(pcl.T, pcl))
    w_max = np.max(w)
    w_min = np.min(w)

    col = np.where(w == w_max)[0]
    if len(col) > 1:
        col = col[-1]
    V_max = v[:, col]

    col_min = np.where(w == w_min)[0]
    if len(col_min) > 1:
        col_min = col_min[-1]
    V_min = v[:, col_min]

    return V_max, V_min, w_max, w_min

def get_grasp(dm, thresh):
    v = abs(np.amin(dm)) - thresh
    if v < 0:
        return 0.0
    elif v > 1:
        return 1.0
    return v

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
    contact_pub = rospy.Publisher('/contact', PoseStamped)
    grasp_pub = rospy.Publisher('/grasped', Float32)

    width = rospy.get_param('~image_size/width')
    height = rospy.get_param('~image_size/height')

    grasp_thresh = rospy.get_param('~grasp_thresh', 4.7)
    depth_max = rospy.get_param('~depth_thresh/max')
    depth_min = rospy.get_param('~depth_thresh/min')

    nn_path = rospy.get_param('~nn_path')
    nn_compute = rospy.get_param('~nn_compute', 'gpu')

    nn = gs3drecon.Reconstruction3D(gs3drecon.Finger.R15)
    nn.load_nn(nn_path, nn_compute)

    #vis3d = gs3drecon.Visualize3D(120, 160, '', 0.0887)

    init_dm = None

    rate = rospy.Rate(30)
    while not rospy.is_shutdown():
        try:
            if use_http and dev.while_condition:
                last_frame = dev.get_image((60, 100, 375, 380))

            if last_frame is not None: 
                frame = last_frame.copy()
                dm = nn.get_depthmap(frame, False)
                dm *= -1
                if init_dm is None:
                    init_dm = dm  

                dm = dm - init_dm 
                dm[dm < depth_min] = 0.0
                dm[dm > depth_max] = 0.0

                pcl = depth2pcl(120, 160, 0.0887, dm)
                v_max, v_min, wx, wy = pca(dm)
                m = np.mean(dm, 0).reshape(-1)
                pose = PoseStamped()
                print(m)
                pose.header.frame_id = "map"
                pose.pose.position.x = m[0]
                pose.pose.position.y = m[1]
                pose.pose.orientation.w = 1
                # pose.theta = math.atan2(v_max[1], v_max[0]) / math.pi * 180

                contact_pub.publish(pose)
                pcl_pub.publish(pcl)
                grasp_pub.publish(Float32(get_grasp(dm, grasp_thresh)))
                #vis3d.update(dm)
                    
            rate.sleep()
        except rospy.ROSInterruptException:
            pass
