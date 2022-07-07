#!/usr/bin/env python3

import gelsight_ros as gsr
from gelsight_ros.msg import GelsightFlowStamped, GelsightMarkersStamped
from geometry_msgs.msg import PoseStamped
import rospy
from sensor_msgs.msg import PointCloud2, Image

# ROS defaults
DEFAULT_RATE = 30
DEFAULT_QUEUE_SIZE = 2
DEFAULT_IMAGE_TOPIC_NAME = "raw"
DEFAULT_INPUT_TYPE = "http_stream"
DEFAULT_DEPTH_METHOD = "poisson"
DEFAULT_DEPTH_TOPIC_NAME = "depth"
DEFAULT_MARKER_TOPIC_NAME = "markers"
DEFAULT_FLOW_TOPIC_NAME = "flow"
DEFAULT_FLOW_IMAGE_TOPIC_NAME = "flow_image"
DEFAULT_POSE_TOPIC_NAME = "pose"

if __name__ == "__main__":
    rospy.init_node("gelsight")
    rate = rospy.Rate(rospy.get_param("~rate", DEFAULT_RATE))
    gelsight_pipeline = [] # List of (GelsightProc, publisher) tuple

    # Configure image stream
    stream = None
    input_type = rospy.get_param("~input_type", DEFAULT_INPUT_TYPE)
    if input_type == "http_stream":
        if not rospy.has_param("~http_stream"):
            rospy.signal_shutdown("No config provided for HTTP stream, but 'http_stream' input selected. Please set http_stream/.")
        http_cfg = rospy.get_param("~http_stream")
        if "url" not in http_cfg or http_cfg["url"] == "":
            rospy.signal_shutdown("URL for GelSight camera stream not provided in config. Please set http_stream/url.")

        if not all(roi in http_cfg for roi in ["roi_x0", "roi_y0", "roi_x1", "roi_y1"]):
            rospy.signal_shutdown("Missing one of the roi corners. Please check http_stream/roi_X.")
        roi = (http_cfg["roi_x0"], http_cfg["roi_y0"], http_cfg["roi_x1"], http_cfg["roi_y1"])
        stream = gsr.GelsightHTTPStream(http_cfg["url"], roi)

        if "publish_image" in http_cfg and http_cfg["publish_image"]:
            image_proc = gsr.ImageProc(stream)
            image_pub = rospy.Publisher(DEFAULT_IMAGE_TOPIC_NAME, Image, queue_size=DEFAULT_QUEUE_SIZE)
            gelsight_pipeline.append((image_proc, image_pub))

    elif input_type == "file_stream":
        if not rospy.has_param("~file_stream/path"):
            rospy.signal_shutdown("No file path provided, but 'file_stream' input selected. Please set file_stream/path.")
        stream = gsr.GelsightFileStream(rospy.get_param("~file_stream/path"))
    else:
        rospy.signal_shutdown(f"Input type not recognized or supported: {input_type}")

    # Load depth reconstruction process
    if rospy.get_param("~depth/enable", False):
        depth_cfg = rospy.get_param("~depth")
        depth_method = rospy.get_param("~depth/method", DEFAULT_DEPTH_METHOD)

        # Compute depth only using poisson approx
        if depth_method == "poisson":
            depth_proc = gsr.DepthFromPoissonProc(stream, depth_cfg)
            topic_name = rospy.get_param("~depth/topic_name", DEFAULT_DEPTH_TOPIC_NAME)
            depth_pub = rospy.Publisher(topic_name, PointCloud2, queue_size=DEFAULT_QUEUE_SIZE)
            gelsight_pipeline.append((depth_proc, depth_pub))

            # Load pose process
            if rospy.get_param("~pose/enable", False):
                pose_cfg = rospy.get_param("~pose")
                pose_proc = gsr.PoseFromDepthProc(depth_proc, pose_cfg)
                topic_name = rospy.get_param("~pose/topic_name", DEFAULT_POSE_TOPIC_NAME)
                pose_pub = rospy.Publisher(topic_name, PoseStamped, queue_size=DEFAULT_QUEUE_SIZE)
                gelsight_pipeline.append((pose_proc, pose_pub))
        
        # Compute depth using neural-network 
        elif depth_method == "nn":
            depth_proc = gsr.DepthFromModelProc(stream, depth_cfg)
            topic_name = rospy.get_param("~depth/topic_name", DEFAULT_DEPTH_TOPIC_NAME)
            depth_pub = rospy.Publisher(topic_name, PointCloud2, queue_size=DEFAULT_QUEUE_SIZE)
            gelsight_pipeline.append((depth_proc, depth_pub))

            # Load pose process
            if rospy.get_param("~pose/enable", False):
                pose_cfg = rospy.get_param("~pose")
                pose_proc = gsr.PoseFromDepthProc(depth_proc, pose_cfg)
                topic_name = rospy.get_param("~pose/topic_name", DEFAULT_POSE_TOPIC_NAME)
                pose_pub = rospy.Publisher(topic_name, PoseStamped, queue_size=DEFAULT_QUEUE_SIZE)
                gelsight_pipeline.append((pose_proc, pose_pub))
        else:
            rospy.logwarn(f"Depth method not recognized or supported: {depth_method}")

    # Load marker process
    if rospy.get_param("~markers/enable", False):
        marker_cfg = rospy.get_param("~markers")
        marker_proc = gsr.MarkersProc(stream, marker_cfg)
        topic_name = rospy.get_param("~markers/topic_name", DEFAULT_MARKER_TOPIC_NAME)
        marker_pub = rospy.Publisher(topic_name, GelsightMarkersStamped, queue_size=DEFAULT_QUEUE_SIZE)
        gelsight_pipeline.append((marker_proc, marker_pub))

        # Load flow process
        if rospy.get_param("~flow/enable", False):
            flow_cfg = rospy.get_param("~flow")
            flow_proc = gsr.FlowProc(marker_proc, flow_cfg)
            topic_name = rospy.get_param("~flow/topic_name", DEFAULT_FLOW_TOPIC_NAME)
            flow_pub = rospy.Publisher(topic_name, GelsightFlowStamped, queue_size=DEFAULT_QUEUE_SIZE)
            gelsight_pipeline.append((flow_proc, flow_pub))

            if rospy.get_param("~flow/publish_image", False):
                flow_im_proc = gsr.DrawFlowProc(stream, flow_proc)
                flow_im_pub = rospy.Publisher(flow_image, Image, queue_size=DEFAULT_QUEUE_SIZE)
                gelsight_pipeline.append((flow_im_proc, flow_im_pub))
    
    elif rospy.get_param("~flow/enable", False):
        rospy.log_warn("Flow detection is enabled, but marker tracking is disabled. Flow will be ignored.")

    # Main loop
    while not rospy.is_shutdown() and stream.while_condition:
        try:
            for proc, pub in gelsight_pipeline:
                try:
                    msg = proc.execute()
                    if msg is not None:
                        if hasattr(msg, "header"):
                            msg.header.frame_id = "map"
                        pub.publish(msg)
                except NotImplementedError:
                    rospy.logwarn(f"Feature not implemented")
                except Exception as e:
                    rospy.logerr(f"Gelsight process failed: {e}")
            rate.sleep()
        except rospy.ROSInterruptException:
            pass
