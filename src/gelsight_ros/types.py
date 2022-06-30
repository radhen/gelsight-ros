#!/usr/bin/env python3

from dataclasses import dataclass
from gelsight_ros.msg import GelsightMarkers as GelsightMarkersMsg, \
    GelsightMarkersStamped as GelsightMarkersStampedMsg, GelsightFlowStamped as GelsightFlowStampedMsg
from geometry_msgs.msg import PoseStamped
import numpy as np
from sensor_msgs import point_cloud2
from sensor_msgs.msg import PointCloud2, PointField
from tf.transformations import quaternion_from_euler

@dataclass
class GelsightMarkers:
    """Stores position of each marker within a frame"""
    rows: int
    cols: int 
    markers: np.ndarray # List of markers coordinates (n_markers, 2)

    def __post_init__(self):
        if self.rows <= 0:
            raise ValueError(f"image width cannot be less than or equal to 0, given: {self.rows}")
        elif self.im_height <= 0:
            raise ValueError(f"image height cannot be less than or equal to 0, given: {self.cols}")
        elif self.markers.shape[1] != 2:
            raise ValueError(f"markers must have shape (n_markers, 2), given shape: {self.markers.shape}")

    def get_ros_msg(self) -> GelsightMarkersMsg:
        raise NotImplementedError()

    def get_ros_msg_stamped(self) -> GelsightMarkersStampedMsg:
        raise NotImplementedError()
    

@dataclass
class GelsightFlow:
    """Measures marker displacement"""
    ref: GelsightMarkers
    cur: GelsightMarkers

    def __post_init__(self):
        if len(self.ref.markers) != len(self.cur.markers):
            raise ValueError(f"Reference and current markers have different sizes")

    def get_ros_msg_stamped(self) -> GelsightFlowStampedMsg:
        raise NotImplementedError()
        # flow_msg = MarkerFlow() flow_msg.n = n
        # flow_msg.m = m
        # for i in range(len(Ox)):
        #     for j in range(len(Ox[i])):
        #         x = MATCHING_SCALE * (Cx[i][j] - Ox[i][j])
        #         y = MATCHING_SCALE * (Cy[i][j] - Ox[i][j])
        #         flow_msg.data.append(Vector3(x=x, y=y))

@dataclass
class GelsightDepth:
    """Stores depth at each pixel"""
    im_width: int
    im_height: int
    depth: np.ndarray # (im_height, im_width, dtype=float32)

    def __post_init__(self):
        if self.depth.shape[0] != self.im_height:
            raise ValueError(f"Number of depth rows ({self.depth.shape[0]}) != set height ({self.im_height})")
        if self.depth.shape[1] != self.im_width:
            raise ValueError(f"Number of depth cols ({self.depth.shape[1]}) != set width ({self.im_width})")

    def get_ros_msg(self, mpp: float) -> PointCloud2:
        points = [] 
        for i in range(self.im_width):
            for j in range(self.im_height):
                points.append(
                    ((i - (self._dm.shape[1]//2)) * mpp,
                     (j - (self._dm.shape[0]//2)) * mpp,
                     self._dm[j, i] / 1000.0, int(0))
                )

        fields = [
            PointField("x", 0, PointField.FLOAT32, 1),
            PointField("y", 4, PointField.FLOAT32, 1),
            PointField("z", 8, PointField.FLOAT32, 1),
            PointField("rgb", 12, PointField.UINT32, 1),
        ]

        header = Header()
        return point_cloud2.create_cloud(header, fields, points)

@dataclass
class GelsightPose:
    """Stores pose of contact"""

    x: float
    y: float
    theta: float

    def get_ros_msg(self) -> PoseStamped:
        pose = PoseStamped()
        pose.pose.position.x = self.x
        pose.pose.position.y = self.y
        pose.pose.position.z = 0.0

        x, y, w, z = quaternion_from_euler(0.0, 0.0, self.theta)
        pose.pose.orientation.x = x
        pose.pose.orientation.y = y
        pose.pose.orientation.z = w
        pose.pose.orientation.w = z

        return pose