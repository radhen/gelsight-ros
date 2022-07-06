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

from .proc import GelsightProc
from .data import GelsightDepth, GelsightFlow, GelsightMarkers, GelsightPose
from .gs3drecon import Reconstruction3D, Finger
from .stream import GelsightStream
from .util import *


class MarkersProc(GelsightProc):
    """
    Reads from stream and returns point of each marker.
    
    execute() -> GelsightMarkersStamped msg 

    Params:
      - threshold_block_size (default: 17)
      - threshold_neg_biad (default: 25)
      - marker_neighborhood_size (default: 20)
    """

    # Parameter defaults
    threshold_block_size: int = 17
    threshold_neg_bias: int = 25
    marker_neighborhood_size: int = 20

    def __init__(self, stream: GelsightStream, cfg: Dict[str, Any]):
        super().__init__()
        self._stream: GelsightStream = stream
        self._markers: Optional[GelsightMarkers] = None

        if "threshold_block_size" in cfg:
            self.threshold_block_size = cfg["threshold_block_size"]
        if "threshold_neg_bias" in cfg:
            self.threshold_neg_bias = cfg["threshold_neg_bias"]
        if "marker_neighborhood_size" in cfg:
            self.marker_neighborhood_size = cfg["marker_neighborhood_size"]

    def execute(self) -> GelsightMarkersStampedMsg:
        # Threshold image to mask markers 
        im = self._stream.get_frame()
        gray_im = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
        im_mask = cv2.adaptiveThreshold(gray_im, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,
            self.threshold_block_size, self.threshold_neg_bias)

        # Find peaks
        max = maximum_filter(im_mask, self.marker_neighborhood_size)
        maxima = im_mask == max
        min = minimum_filter(im_mask, self.marker_neighborhood_size)
        diff = (max - min) > 1
        maxima[diff == 0] = 0

        labeled, n = ndimage.label(maxima)
        xy = np.array(ndimage.center_of_mass(im_mask, labeled, range(1, n + 1)))
        xy[:, [0, 1]] = xy[:, [1, 0]]

        # Convert to gelsight dataclass
        self._markers = GelsightMarkers(im.shape[1], im.shape[0], xy)
        return self._markers.get_ros_msg_stamped()

    def get_markers(self) -> Optional[GelsightMarkers]:
        return self._markers


class FlowProc(GelsightProc):
    """
    Approximates displacement of markers. 

    execute() -> GelsightFlowStamped msg

    Params:
      - x0 (Required)
      - y0 (Required)
      - dx (Required)
      - dy (Required)
      - marker_shape (default: (10, 12))
      - matching_fps (default: 25)
      - flow_scale (default: 5)
    """


    # Parameter defaults
    marker_shape: Tuple[int, int] = (10, 12) 
    matching_fps: int = 25
    flow_scale: float = 5

    def __init__(self, markers: MarkersProc, cfg: Dict[str, Any]):
        super().__init__()
        self._markers: MarkersProc = markers
        self._flow: Optional[GelsightFlow] = None

        if not all(param in cfg for param in ["x0", "y0", "dx", "dy"]):
            raise RuntimeError("FlowProc: Missing marker configuration.")

        if "flow_scale" in cfg:
            self.flow_scale = cfg["flow_scale"]

        self._match = Matching(
            self.marker_shape[0], self.marker_shape[1],
            self.matching_fps, cfg["x0"], cfg["y0"], cfg["dx"], cfg["dy"]
        )

    def execute(self) -> GelsightFlowStampedMsg:
        gsmarkers = self._markers.get_markers()  
        if gsmarkers: 
            self._match.init(gsmarkers.markers)
            
            self._match.run()
            Ox, Oy, Cx, Cy, _ = self._match.get_flow()

            # Transform into shape: (n_markers, 2)
            Ox_t = np.reshape(np.array(Ox).flatten(), (len(Ox) * len(Ox[0]), 1))
            Oy_t = np.reshape(np.array(Oy).flatten(), (len(Oy) * len(Oy[0]), 1))
            ref_markers = GelsightMarkers(len(Oy), len(Ox), np.hstack((Ox_t, Oy_t)))
            Cx_t = np.reshape(np.array(Cx).flatten(), (len(Cx) * len(Cx[0]), 1))
            Cy_t = np.reshape(np.array(Cy).flatten(), (len(Cy) * len(Cy[0]), 1))
            cur_markers = GelsightMarkers(len(Cy), len(Cx), np.hstack((Cx_t, Cy_t)))

            self._flow = GelsightFlow(ref_markers, cur_markers)

            return self._flow.get_ros_msg_stamped()

    def get_flow(self) -> Optional[GelsightFlow]:
        return self._flow


class DrawFlowProc(GelsightProc):
    """
    Reads from stream and flow, then returns image with flow.
    
    execute() -> Image msg 
    """

    def __init__(self, stream: GelsightStream, flow: FlowProc, cfg: Dict[str, Any]):
        super().__init__()
        self._stream: GelsightStream = stream
        self._markers: Optional[GelsightMarkers] = None

        if "threshold_block_size" in cfg:
            self.threshold_block_size = cfg["threshold_block_size"]
        if "threshold_neg_bias" in cfg:
            self.threshold_neg_bias = cfg["threshold_neg_bias"]
        if "marker_neighborhood_size" in cfg:
            self.marker_neighborhood_size = cfg["marker_neighborhood_size"]

    def execute(self) -> GelsightMarkersStampedMsg:
        # Threshold image to mask markers 
        im = self._stream.get_frame()
        gray_im = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
        im_mask = cv2.adaptiveThreshold(gray_im, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,
            self.threshold_block_size, self.threshold_neg_bias)

        # Find peaks
        max = maximum_filter(im_mask, self.marker_neighborhood_size)
        maxima = im_mask == max
        min = minimum_filter(im_mask, self.marker_neighborhood_size)
        diff = (max - min) > 1
        maxima[diff == 0] = 0

        labeled, n = ndimage.label(maxima)
        xy = np.array(ndimage.center_of_mass(im_mask, labeled, range(1, n + 1)))
        xy[:, [0, 1]] = xy[:, [1, 0]]

        # Convert to gelsight dataclass
        self._markers = GelsightMarkers(im.shape[1], im.shape[0], xy)
        return self._markers.get_ros_msg_stamped()

    def get_markers(self) -> Optional[GelsightMarkers]:
        return self._markers