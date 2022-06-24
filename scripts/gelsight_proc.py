#!/usr/bin/env python3

import rospy
import math
import cv2
import numpy as np
np.random.BitGenerator = np.random.bit_generator.BitGenerator
from numpy import linalg as LA
from collections import deque
from enum import Enum

from std_msgs.msg import Header, Float32
from sensor_msgs import point_cloud2
from sensor_msgs.msg import Image, PointCloud2, PointField
from geometry_msgs.msg import PoseStamped
from tf.transformations import quaternion_from_euler
from cv_bridge import CvBridge
from gelsight_ros.msg import MarkerFlow

from gelsight import gsdevice
from gelsight import gs3drecon

from gelsight_ros.util import *


class ThreshType(Enum):
    GAUSSIAN = 1
    EXPONENTIAL = 2


last_frame = None
bridge = CvBridge()


def image_cb(msg):
    global last_frame
    last_frame = bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")


if __name__ == "__main__":
    rospy.init_node("gelsight_proc")

    frame_id = rospy.get_param("~frame_id", "map")
    rate = rospy.Rate(rospy.get_param("~rate", 30))

    use_http = rospy.get_param("~use_http", True)
    cam_url = rospy.get_param("~cam_url", "")
    if use_http and cam_url == "":
        rospy.logfatal('No "cam_url" provided when "use_http" is true')

    width = rospy.get_param("~image_size/width")
    height = rospy.get_param("~image_size/height")
    roi = (
        rospy.get_param("~image_roi/top_left/x"),
        rospy.get_param("~image_roi/top_left/y"),
        rospy.get_param("~image_roi/bottom_right/x"),
        rospy.get_param("~image_roi/bottom_right/y"),
    )

    depth_thresh_type = rospy.get_param("~depth_thresh/type", None)
    gauss_params = rospy.get_param("~depth_thresh/gaussian", None)
    exp_params = rospy.get_param("~depth_thresh/exponential", None)
    if depth_thresh_type is not None:
        if depth_thresh_type == "gaussian":
            depth_thresh_type = ThreshType.GAUSSIAN
        elif depth_thresh_type == "exponential":
            depth_thresh_type = ThreshType.EXPONENTIAL
        else:
            rospy.logfatal(f"Depth thresh type not recognized: {depth_thresh_type}")

    nn_path = rospy.get_param("~nn_path")
    nn_compute = rospy.get_param("~nn_compute", "gpu")
    nn_output_width = rospy.get_param("~nn_output_size/width")
    nn_output_length = rospy.get_param("~nn_output_size/height")
    nn_mmpp = rospy.get_param("~nn_mmpp")

    publish_markers = rospy.get_param("~publish_markers", False)
    n_markers = rospy.get_param("~n_markers")
    m_markers = rospy.get_param("~m_markers")

    gaussian_width = rospy.get_param("~gaussian_kernel/width")
    gaussian_height = rospy.get_param("~gaussian_kernel/height")

    pcl_pub = rospy.Publisher("/pcl", PointCloud2, queue_size=1)
    contact_pub = rospy.Publisher("/contact", PoseStamped)
    grasp_pub = rospy.Publisher("/grasped", Float32)
    marker_flow_pub = rospy.Publisher("/flow", MarkerFlow)

    if not use_http:
        image_sub = rospy.Subscriber("/image/raw", Image, image_cb, queue_size=1)
    else:
        dev = gsdevice.Camera(gsdevice.Finger.R15, cam_url)
        dev.connect()

    nn = gs3drecon.Reconstruction3D(gs3drecon.Finger.R15)
    nn.load_nn(nn_path, nn_compute)

    init_frame = None
    init_markers = None
    init_dm = None
    pca_buffer = deque([], maxlen=rospy.get_param("~pca_buffer_size"))
    while not rospy.is_shutdown():
        try:
            if use_http and dev.while_condition:
                last_frame = dev.get_image(roi)

            if last_frame is not None:
                frame = last_frame.copy()

                if init_frame is None:
                    init_frame = frame

                if publish_markers:
                    if init_markers is None:
                        init_markers = image2markers(frame)
                    else:
                        flow_msg = image2flow(init_markers, init_frame, frame, n_markers, m_markers)
                        marker_flow_pub.publish(flow_msg)

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
                        gauss = get_2d_gaussian(
                            dm.shape[0], dm.shape[1], gauss_params["sig"]
                        )
                        thresh = dm - gauss
                        dm[thresh > gauss_params["max"]] = 0.0
                        dm[thresh < gauss_params["min"]] = 0.0
                    elif depth_thresh_type == ThreshType.EXPONENTIAL:
                        exp = get_2d_exponential(
                            dm.shape[0], dm.shape[1], exp_params["beta"]
                        )
                        thresh = dm - exp
                        dm[thresh > exp_params["max"]] = 0.0
                        dm[thresh < exp_params["min"]] = 0.0

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
