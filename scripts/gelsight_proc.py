#!/usr/bin/env python3

import rospy
import math
import cv2
import numpy as np
from numpy import linalg as LA
from collections import deque
from enum import Enum

from std_msgs.msg import Header, Float32
from sensor_msgs import point_cloud2
from sensor_msgs.msg import Image, PointCloud2, PointField
from geometry_msgs.msg import PoseStamped
from tf.transformations import quaternion_from_euler
from cv_bridge import CvBridge

from gelsight import gsdevice
from gelsight import gs3drecon

class ThreshType(Enum):
    GAUSSIAN = 1
    EXPONENTIAL = 2

last_frame = None
bridge = CvBridge()

def image_cb(msg):
    global last_frame
    last_frame = bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')

def depth2pcl(width, length, mmpp, dm):
    points = []
    for i in range(width):
        for j in range(length):
            points.append((i*mmpp/100.0, j*mmpp/100.0, dm[j, i]/1000.0, int(0)))


    fields = [PointField('x', 0, PointField.FLOAT32, 1),
              PointField('y', 4, PointField.FLOAT32, 1),
              PointField('z', 8, PointField.FLOAT32, 1),
              PointField('rgb', 12, PointField.UINT32, 1)]

    header = Header()
    return point_cloud2.create_cloud(header, fields, points)

def depth2pca(dm, mmpp, buffer):
    pnts = np.where(dm > 0)
    X = pnts[1].reshape(-1, 1)
    Y = pnts[0].reshape(-1, 1)
    pnts = np.concatenate([X, Y], axis=1)
    pnts = pnts.reshape(-1, 2).astype(np.float64) 
    if pnts.shape[0] == 0 :
        return None
    
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

    if len(buffer) > 0:
        buffer.popleft() 
    buffer.append((mv[0], mv[1], theta))

    x_bar = 0.0
    y_bar = 0.0
    theta_bar = 0.0
    for a in list(buffer):
      x, y, theta = a
      x_bar += x
      y_bar += y
      theta_bar += theta

    if len(buffer) > 0:
      x_bar /= len(buffer)
      y_bar /= len(buffer)
      theta_bar /= len(buffer) 

    pose = PoseStamped()
    pose.pose.position.x = x_bar*mmpp/100
    pose.pose.position.y = y_bar*mmpp/100
    pose.pose.position.z = 0.0 
    
    x, y, w, z = quaternion_from_euler(0.0, 0.0, theta_bar)
    pose.pose.orientation.x = x
    pose.pose.orientation.y = y
    pose.pose.orientation.z = w
    pose.pose.orientation.w = z
    
    return pose

def get_2d_gaussian(n, m, sig):
    x = np.linspace(-sig, sig, n)
    x_gauss = np.exp(-0.5 * np.square(x) / np.square(sig))

    y = np.linspace(-sig, sig, m)
    y_gauss = np.exp(-0.5 * np.square(y) / np.square(sig))

    return np.outer(x_gauss, y_gauss)

def get_2d_exponential(n, m, beta):
    x = np.linspace(0.0, beta, n//2) 
    if n%2 == 0:
        x = np.concatenate([x, np.linspace(beta, 0.0, n//2)])
    else:
        x = np.concatenate([x, np.linspace(beta, 0.0, (n//2)+1)])

    y = np.linspace(0.0, beta, m//2) 
    if m%2 == 0:
        y = np.concatenate([y, np.linspace(beta, 0.0, m//2)])
    else:
        y = np.concatenate([y, np.linspace(beta, 0.0, (m//2)+1)])
    
    return np.outer(np.exp(x), np.exp(y))

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

    depth_thresh_type = rospy.get_param('~depth_thresh/type', None) 
    gauss_params = rospy.get_param('~depth_thresh/gaussian', None)
    exp_params = rospy.get_param('~depth_thresh/exponential', None)
    if depth_thresh_type is not None:
        if depth_thresh_type == "gaussian":
            depth_thresh_type = ThreshType.GAUSSIAN
        elif depth_thresh_type == "exponential":
            depth_thresh_type = ThreshType.EXPONENTIAL
        else:
            rospy.logfatal(f'Depth thresh type not recognized: {depth_thresh_type}')

    nn_path = rospy.get_param('~nn_path')
    nn_compute = rospy.get_param('~nn_compute', 'gpu')
    nn_output_width = rospy.get_param('~nn_output_size/width')
    nn_output_length = rospy.get_param('~nn_output_size/height')
    nn_mmpp = rospy.get_param('~nn_mmpp')

    gaussian_width = rospy.get_param('~gaussian_kernel/width')
    gaussian_height = rospy.get_param('~gaussian_kernel/height')

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
    pca_buffer = deque([], maxlen=rospy.get_param('~pca_buffer_size'))
    while not rospy.is_shutdown():
        try:
            if use_http and dev.while_condition:
                last_frame = dev.get_image(roi)

            if last_frame is not None: 
                frame = last_frame.copy()
                
                dm = nn.get_depthmap(frame, False)

                dm *= -1
                if init_dm is None:
                    init_dm = dm  

                dm = dm - init_dm 

                pcl = depth2pcl(nn_output_width, nn_output_length, nn_mmpp, dm)
                pcl.header.frame_id = frame_id
                pcl_pub.publish(pcl)

                if depth_thresh_type is not None:
                    if depth_thresh_type == ThreshType.GAUSSIAN:
                        gauss = get_2d_gaussian(dm.shape[0], dm.shape[1], gauss_params['sig'])
                        thresh = dm - gauss
                        dm[thresh > gauss_params['max']] = 0.0
                        dm[thresh < gauss_params['min']] = 0.0
                    elif depth_thresh_type == ThreshType.EXPONENTIAL:
                        exp = get_2d_exponential(dm.shape[0], dm.shape[1], exp_params['beta'])
                        thresh = dm - exp
                        dm[thresh > exp_params['max']] = 0.0
                        dm[thresh < exp_params['min']] = 0.0

                pose = depth2pca(dm, nn_mmpp, pca_buffer)
                if pose is not None:
                    pose.header.frame_id = frame_id
                    contact_pub.publish(pose)

                    grasp_pub.publish(Float32(1.0))
                else:
                    grasp_pub.publish(Float32(0.0))

            rate.sleep()
        except rospy.ROSInterruptException:
            pass
