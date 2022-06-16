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
from tf.transformations import quaternion_from_euler

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
    return point_cloud2.create_cloud(header, fields, points)

def depth2pca(dm, mmpp):
    pnts = np.where(dm > 0)
    X = pnts[1].reshape(-1, 1)
    Y = pnts[0].reshape(-1, 1)
    pnts = np.concatenate([X, Y], axis=1)
    pnts = pnts.reshape(-1, 2).astype(np.float64)
    mv = np.mean(pnts, 0).reshape(2, 1)
    pnts -= mv.T
    w, v = LA.eig(np.dot(pnts.T, pnts))
    w_max = np.max(w)

    col = np.where(w == w_max)[0]
    if len(col) > 1:
        col = col[-1]
     
    V_max = v[:, col]
    if V_max[0] > 0 and V_max[1] > 0:
        V_max *= -1
    
    V_max = V_max.reshape(-1) * (w_max ** 0.3 / 1)
    theta = math.atan2(V_max[1], V_max[0]) 

    pose = PoseStamped()
    pose.pose.position.x = mv[0]*mmpp/100
    pose.pose.position.y = mv[1]*mmpp/100
    pose.pose.position.z = 0.0 
    
    x, y, w, z = quaternion_from_euler(0.0, 0.0, theta)
    pose.pose.orientation.x = x
    pose.pose.orientation.y = y
    pose.pose.orientation.z = w
    pose.pose.orientation.w = z
    
    return pose

def get_grasp_score(dm, thresh):
    v = abs(np.amin(dm)) - thresh
    if v < 0:
        return 0.0
    elif v > 1:
        return 1.0
    return v

if __name__ == '__main__':
    rospy.init_node('gelsight_proc')

    frame_id = rospy.get_param('~frame_id', 'map')
    rate = rospy.Rate(rospy.get_param('~rate', 30))
    
    use_http = rospy.get_param('~use_http', True)
    cam_url = rospy.get_param('~cam_url', '')
    if use_http and cam_url == '':
        rospy.logfatal('No "cam_url" provided when "use_http" is true')

    width = rospy.get_param('~image_size/width')
    height = rospy.get_param('~image_size/height')
    roi = (rospy.get_param('~image_roi/top_left/x'),
           rospy.get_param('~image_roi/top_left/y'),
           rospy.get_param('~image_roi/bottom_right/x'),
           rospy.get_param('~image_roi/bottom_right/y'))

    grasp_thresh = rospy.get_param('~grasp_thresh', 4.7)
    depth_max = rospy.get_param('~depth_thresh/max')
    depth_min = rospy.get_param('~depth_thresh/min')

    nn_path = rospy.get_param('~nn_path')
    nn_compute = rospy.get_param('~nn_compute', 'gpu')

    pcl_pub = rospy.Publisher('/pcl', PointCloud2, queue_size=1)
    contact_pub = rospy.Publisher('/contact', PoseStamped)
    grasp_pub = rospy.Publisher('/grasped', Float32)

    if not use_http: 
        image_sub = rospy.Subscriber('/image/raw', Image, image_cb, queue_size=1)
    else:
        dev = gsdevice.Camera(gsdevice.Finger.R15, cam_url)
        dev.connect()

    nn = gs3drecon.Reconstruction3D(gs3drecon.Finger.R15)
    nn.load_nn(nn_path, nn_compute)

    init_dm = None
    while not rospy.is_shutdown():
        try:
            if use_http and dev.while_condition:
                last_frame = dev.get_image(roi)

            if last_frame is not None: 
                frame = last_frame.copy()
                
                dm = nn.get_depthmap(frame, False)

                pcl = depth2pcl(120, 160, 0.04, dm)
                pcl.header.frame_id = frame_id
                pcl_pub.publish(pcl)
                
                dm *= -1
                if init_dm is None:
                    init_dm = dm  
		
                dm = dm - init_dm 
                dm = cv2.GaussianBlur(dm, (13, 13), 0)    
                dm[dm < depth_min] = 0.0
                dm[dm > depth_max] = 0.0

                pose = depth2pca(dm, 0.0887)
                pose.header.frame_id = frame_id
                contact_pub.publish(pose)

                grasp_pub.publish(Float32(get_grasp_score(dm, grasp_thresh)))
                    
            rate.sleep()
        except rospy.ROSInterruptException:
            pass
