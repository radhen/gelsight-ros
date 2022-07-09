#!/usr/bin/env python3

"""
Collects a sequence of images from the sensor.

Stream configured in config/gelsight.yml
"""

import cv2
import gelsight_ros as gsr
import os
import rospy

DEFAULT_RATE = 30
DEFAULT_DURATION = 30

if __name__ == "__main__":
    rospy.init_node("record")
    rate = rospy.Rate(rospy.get_param("~rate", DEFAULT_RATE))
    end_time = rospy.Time.now() + rospy.Duration(rospy.get_param("~num_secs", DEFAULT_DURATION))
    
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

    cfg = rospy.get_param("~http_stream")
    if not cfg:
        rospy.signal_shutdown("No config provided for HTTP stream. Please set http_stream/.")

    if "url" not in cfg:
        rospy.signal_shutdown("No URL provided for HTTP stream. Please set http_stream/url.")

    if not all(roi in cfg for roi in ["roi_x0", "roi_y0", "roi_x1", "roi_y1"]):
        rospy.signal_shutdown("Missing one of the roi corners. Please check http_stream/roi_X.")
    roi = (cfg["roi_x0"], cfg["roi_y0"], cfg["roi_x1"], cfg["roi_y1"])

    stream = gsr.GelsightHTTPStream(cfg["url"], roi)
    
    # Main loop
    i = 0
    while not rospy.is_shutdown() and stream.while_condition and rospy.Time.now() < end_time:
        try:
            frame = stream.get_frame()
            if not cv2.imwrite(f"{output_path}/{i}.jpg", frame):
                rospy.logwarn(f"Failed to write file to {output_path}")
            else:
                i += 1
            rate.sleep()
        except rospy.ROSInterruptException:
            pass
