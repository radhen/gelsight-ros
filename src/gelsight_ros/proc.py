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

class DepthProc(GelsightProc):
    """Converts image streams into depth maps."""

    def get_depth(self) -> Optional[GelsightDepth]:
        raise NotImplementedError()

    def get_mpp(self) -> float:
        raise NotImplementedError()

class DepthFromModelProc(DepthProc):
    """
    Uses trained NN to convert image streams into depth maps.
    
    execute() -> PointCloud2 msg 

    Params:
      - model_path (Required)
      - compute_type (default: 'cpu')
      - model_output_width (default: 120)
      - model_output_height (default: 160)
    """

    # Parameter defaults
    compute_type: str = "cpu"
    image_mpp: float = 0.005
    model_output_width: int = 120
    model_output_height: int = 160

    def __init__(self, stream: GelsightStream, cfg: Dict[str, Any]):
        super().__init__()
        self._stream: GelsightStream = stream

        if "model_path" not in cfg:
            raise RuntimeError("DepthProc: Missing model path.")

        if "compute_type" in cfg:
            self.compute_type = cfg["compute_type"]
        if "image_mpp" in cfg:
            self.image_mpp = cfg["image_mpp"]
        if "model_output_width" in cfg:
            self.model_output_width = cfg["model_output_width"]
        if "model_output_height" in cfg:
            self.model_output_height = cfg["model_output_height"]

        self._model = Reconstruction3D(Finger.R15)
        self._model.load_nn(cfg["model_path"], self.compute_type)

        self._init_dm: Optional[GelsightDepth] = None
        self._dm: Optional[GelsightDepth] = None

    def execute(self) -> PointCloud2:
        # Compute depth map 
        dm = self._model.get_depthmap(self._stream.get_frame(), False)
        dm *= -1
        if self._init_dm is None:
            self._init_dm = dm
        dm -= self._init_dm

        self._dm = GelsightDepth(self.model_output_width, self.model_output_height, dm)
        return self._dm.get_ros_msg(self.image_mpp)
    
    def get_depth(self) -> Optional[GelsightDepth]:
        return self._dm

    def get_mpp(self) -> float:
        return self.image_mpp


class DepthFromPoissonProc(DepthProc):
    """
    Approximates depth maps from stream using poisson reconstruction.
    See more: https://hhoppe.com/poissonrecon.pdf

    execute() -> PointCloud2 msg

    Params:
      - image_mpp (default: 0.005)
      - image_width (default: 120)
      - image_height (default: 160)
    """

    # Parameter defaults
    image_mpp: float = 0.005
    image_width: int = 120
    image_height: int = 160

    def __init__(self, stream: GelsightStream, cfg: Dict[str, Any]):
        super().__init__()
        self._stream = stream

        if "image_mpp" in cfg:
            self.image_mpp = cfg["image_mpp"]

        self._init_frame = None
        self._init_dm = None

    def img2depth(self, frame: np.ndarray):
        diff = frame * 1.0 - self._init_frame
        dx = diff[:, :, 1] / 255.0
        dx = dx / (1.0 - dx ** 2) ** 0.5 / 32.0
        dy = (diff[:, :, 0] - diff[:, :, 2]) / 255.0
        dy = dy / (1.0 - dy ** 2) ** 0.5 / 32.0

        diff /= 255.0
        dx, dy = demark(frame, dx, dy)

        zeros = np.zeros_like(dx)
        return poisson_reconstruct(dy, dx, zeros)

    def execute(self) -> PointCloud2:
        frame = self._stream.get_frame()
        if self._init_frame is None:
            self._init_frame = frame

        # Compute depth map 
        dm = self.img2depth(frame)        
        dm *= -1
        if self._init_dm is None:
            self._init_dm = dm
        # dm -= self._init_dm

        self._dm = GelsightDepth(self.image_width, self.image_height, dm)
        return self._dm.get_ros_msg(self.image_mpp)
    
    def get_depth(self) -> Optional[GelsightDepth]:
        return self._dm

    def get_mpp(self) -> float:
        return self.image_mpp

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

class PoseFromDepthProc(GelsightProc):
    """
    Approximates contact pose from depth using PCA. 

    execute() -> PoseStamped msg

    Params:
      - buffer_size (default: 5)
    """

    # Parameter defaults
    buffer_size: int = 5

    def __init__(self, depth: DepthProc, cfg: Dict[str, Any]):
        super().__init__()
        self._depth: DepthProc = depth

        if "buffer_size" in cfg:
            self.buffer_size = cfg["buffer_size"]

        self._buffer: deque = deque([], maxlen=self.buffer_size)
        self._pose: Optional[GelsightPose] = None

    def execute(self) -> PoseStamped:
        gsdepth = self._depth.get_depth()

        if gsdepth: 
            dm = gsdepth.depth

            pnts = np.where(dm > 0)
            X = pnts[1].reshape(-1, 1)
            Y = pnts[0].reshape(-1, 1)
            pnts = np.concatenate([X, Y], axis=1)
            pnts = pnts.reshape(-1, 2).astype(np.float64)
            if pnts.shape[0] == 0:
                return None

            mv = np.mean(pnts, 0).reshape(2, 1)
            pnts -= mv.T
            w, v = np.linalg.eig(np.dot(pnts.T, pnts))
            w_max = np.max(w)

            col = np.where(w == w_max)[0]
            if len(col) > 1:
                col = col[-1]

            V_max = v[:, col]
            if V_max[0] > 0 and V_max[1] > 0:
                V_max *= -1

            V_max = V_max.reshape(-1) * (w_max**0.3 / 1)
            theta = math.atan2(V_max[1], V_max[0])

            if len(self._buffer) > 0:
                self._buffer.popleft()
            self._buffer.append((mv[0], mv[1], theta))

            x_bar = 0.0
            y_bar = 0.0
            theta_bar = 0.0
            for a in list(self._buffer):
                x, y, theta = a
                x_bar += x
                y_bar += y
                theta_bar += theta

            if len(self._buffer) > 0:
                x_bar /= len(self._buffer)
                y_bar /= len(self._buffer)
                theta_bar /= len(self._buffer)

            x_bar = (x_bar - (dm.shape[0]//2)) * self._depth.get_mpp()
            y_bar = (y_bar - (dm.shape[1]//2)) * self._depth.get_mpp()
            self._pose = GelsightPose(x_bar, y_bar, theta)
            return self._pose.get_ros_msg()

    def get_pose(self) -> Optional[GelsightPose]:
        return self._pose
