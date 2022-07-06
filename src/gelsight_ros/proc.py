#!/usr/bin/env python3

from collections import deque
import cv2
from cv_bridge import CvBridge, CvBridgeError
from find_marker import Matching
from gelsight_ros.msg import GelsightMarkersStamped as GelsightMarkersStampedMsg, \
    GelsightFlowStamped as GelsightFlowStampedMsg
from geometry_msgs.msg import PoseStamped
import math
import numpy as np
from rospy import AnyMsg
from scipy import ndimage
from scipy.ndimage.filters import maximum_filter, minimum_filter
from sensor_msgs.msg import PointCloud2, Image
from typing import Dict, Tuple, Any, Optional

from .data import GelsightDepth, GelsightFlow, GelsightMarkers, GelsightPose
from .gs3drecon import Reconstruction3D, Finger
from .stream import GelsightStream
from .util import *

class GelsightProc:
    def execute(self) -> AnyMsg:
        raise NotImplementedError()

class ImageProc(GelsightProc):
    """
    Converts stream to sensor msgs.

    execute() -> Image msg
    """

    # Parameter defaults
    encoding = "bgr8"

    def __init__(self, stream: GelsightStream):
        super().__init__()
        self._stream: GelsightStream = stream

    def execute(self) -> Image:
        frame = self._stream.get_frame()
        return CvBridge().cv2_to_imgmsg(frame, self.encoding)
