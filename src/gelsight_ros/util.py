#!/usr/bin/env python3

import cv2
from scipy.signal import fftconvolve
from scipy import ndimage
from scipy.ndimage.filters import maximum_filter, minimum_filter
from math import sqrt
import numpy as np
from sensor_msgs.msg import Image, PointCloud2, PointField
from gelsight_ros.msg import MarkerFlow
from geometry_msgs.msg import Vector3
from std_msgs.msg import Header, Float32
from geometry_msgs.msg import PoseStamped
from tf.transformations import quaternion_from_euler
from sensor_msgs import point_cloud2
from numpy import linalg as LA
import math
from find_marker import Matching

MARKER_INTENSITY_SCALE = 3
MARKER_THRESHOLD = 255
MARKER_TEMPLATE_SIZE = 5
MARKER_TEMPLATE_RADIUS = 3
MARKER_NEIGHBORHOOD_SIZE = 20
MATCHING_FPS = 10
MATCHING_SCALE = 5

def image2markers(image):
    # Mask markers
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # mask = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 9, 25)

    # Gelsight Example Approach 
    scaled = cv2.convertScaleAbs(gray, alpha=MARKER_INTENSITY_SCALE, beta=0)
    mask = cv2.inRange(scaled, MARKER_THRESHOLD, MARKER_THRESHOLD)

    # Perform normalized cross-correlation with gaussian kernel
    # See https://github.com/gelsightinc/gsrobotics/blob/main/examples/markertracking.py
    t = get_2d_gaussian(
        MARKER_TEMPLATE_SIZE, MARKER_TEMPLATE_SIZE, MARKER_TEMPLATE_RADIUS
    )
    t = t - np.mean(t)
    a = mask - np.mean(mask)
    n_xcorr = fftconvolve(a, np.flipud(np.fliplr(t)).conj(), mode="same")
    a = fftconvolve(np.square(a), np.ones(t.shape), mode="same") - np.square(
        fftconvolve(a, np.ones(t.shape), mode="same") / np.prod(t.shape)
    )
    a[np.where(a < 0)] = 0
    t = np.sum(np.square(t))
    n_xcorr = n_xcorr / np.sqrt(a * t)
    n_xcorr[np.where(np.logical_not(np.isfinite(n_xcorr)))] = 0

    # Dilate image
    dilated = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=1)
    b = 2 * ((n_xcorr - n_xcorr.min()) / (n_xcorr.max() - n_xcorr.min())) - 1
    b = (b - b.min()) / (b.max() - b.min())
    mask = np.asarray(b < 0.5)
    mask = (mask * 255).astype("uint8")

    # Find peaks
    max = maximum_filter(mask, MARKER_NEIGHBORHOOD_SIZE)
    maxima = mask == max
    min = minimum_filter(mask, MARKER_NEIGHBORHOOD_SIZE)
    diff = (max - min) > 1
    maxima[diff == 0] = 0

    labeled, n = ndimage.label(maxima)
    xy = np.array(ndimage.center_of_mass(mask, labeled, range(1, n + 1)))
    xy[:, [0, 1]] = xy[:, [1, 0]]
    return xy


def image2flow(markers, n, m, p0, dp):
    match = Matching(m, n, MATCHING_FPS, p0[0], p0[1], dp[0], dp[1])

    match.init(markers)
    match.run()

    Ox, Oy, Cx, Cy, _ = match.get_flow()

    flow_msg = MarkerFlow()
    flow_msg.n = n
    flow_msg.m = m
    for i in range(len(Ox)):
        for j in range(len(Ox[i])):
            x = K * (Cx[i][j] - Ox[i][j])
            y = K * (Cy[i][j] - Ox[i][j])
            flow_msg.data.append(Vector3(x=x, y=y))

    return flow_msg


def depth2pcl(width, length, mmpp, dm):
    points = []
    for i in range(width):
        for j in range(length):
            points.append(
                (i * mmpp / 100.0, j * mmpp / 100.0, dm[j, i] / 1000.0, int(0))
            )

    fields = [
        PointField("x", 0, PointField.FLOAT32, 1),
        PointField("y", 4, PointField.FLOAT32, 1),
        PointField("z", 8, PointField.FLOAT32, 1),
        PointField("rgb", 12, PointField.UINT32, 1),
    ]

    header = Header()
    return point_cloud2.create_cloud(header, fields, points)


def depth2pca(dm, mmpp, buffer):
    pnts = np.where(dm > 0)
    X = pnts[1].reshape(-1, 1)
    Y = pnts[0].reshape(-1, 1)
    pnts = np.concatenate([X, Y], axis=1)
    pnts = pnts.reshape(-1, 2).astype(np.float64)
    if pnts.shape[0] == 0:
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

    V_max = V_max.reshape(-1) * (w_max**0.3 / 1)
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
    pose.pose.position.x = x_bar * mmpp / 100
    pose.pose.position.y = y_bar * mmpp / 100
    pose.pose.position.z = 0.0

    x, y, w, z = quaternion_from_euler(0.0, 0.0, theta_bar)
    pose.pose.orientation.x = x
    pose.pose.orientation.y = y
    pose.pose.orientation.z = w
    pose.pose.orientation.w = z

    return pose


def get_2d_gaussian(n, m, sig):
    x = np.linspace(-(n - 1) / 2.0, (n - 1) / 2.0, n)
    x_gauss = np.exp(-0.5 * np.square(x) / np.square(sig))

    y = np.linspace(-(m - 1) / 2.0, (m - 1) / 2.0, m)
    y_gauss = np.exp(-0.5 * np.square(y) / np.square(sig))

    gauss = np.outer(x_gauss, y_gauss)
    return gauss / np.sum(gauss)


def get_2d_exponential(n, m, beta):
    x = np.linspace(0.0, beta, n // 2)
    if n % 2 == 0:
        x = np.concatenate([x, np.linspace(beta, 0.0, n // 2)])
    else:
        x = np.concatenate([x, np.linspace(beta, 0.0, (n // 2) + 1)])

    y = np.linspace(0.0, beta, m // 2)
    if m % 2 == 0:
        y = np.concatenate([y, np.linspace(beta, 0.0, m // 2)])
    else:
        y = np.concatenate([y, np.linspace(beta, 0.0, (m // 2) + 1)])

    return np.outer(np.exp(x), np.exp(y))


def get_grasp_score(dm, thresh):
    v = abs(np.amin(dm)) - thresh
    if v < 0:
        return 0.0
    elif v > 1:
        return 1.0
    return v
