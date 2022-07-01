#!/usr/bin/env python3

"""
Most of these util functions are copied from Gelsight SDK.

URL: https://github.com/gelsightinc/gsrobotics
"""

import math
import cv2
import numpy as np
import scipy, scipy.fftpack
from scipy.interpolate import griddata

def interpolate_grad(img, mask):
    # mask = (soft_mask > 0.5).astype(np.uint8) * 255
    # cv2.imshow("mask_hard", mask)
    # pixel around markers
    mask_around = (dilate(mask, ksize=3) > 0) & (mask != 1)
    # mask_around = mask == 0
    mask_around = mask_around.astype(np.uint8)

    x, y = np.arange(img.shape[0]), np.arange(img.shape[1])
    yy, xx = np.meshgrid(y, x)

    # mask_zero = mask == 0
    mask_zero = mask_around == 1
    mask_x = xx[mask_zero]
    mask_y = yy[mask_zero]
    points = np.vstack([mask_x, mask_y]).T
    values = img[mask_x, mask_y]
    markers_points = np.vstack([xx[mask != 0], yy[mask != 0]]).T
    method = "nearest"
    # method = "linear"
    # method = "cubic"
    x_interp = griddata(points, values, markers_points, method=method)
    x_interp[x_interp != x_interp] = 0.0
    ret = img.copy()
    ret[mask != 0] = x_interp
    return ret

def dilate(img, ksize=5):
    kernel = np.ones((ksize, ksize), np.uint8)
    return cv2.dilate(img, kernel, iterations=1)


def erode(img, ksize=5):
    kernel = np.ones((ksize, ksize), np.uint8)
    return cv2.erode(img, kernel, iterations=1)

def demark(img, gx, gy):

    gray_im = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    im_mask = cv2.adaptiveThreshold(gray_im, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,
        17, 25)

    gx = interpolate_grad(gx, im_mask)
    gy = interpolate_grad(gy, im_mask)

    return gx, gy

def poisson_reconstruct(grady, gradx, boundarysrc):
    # Thanks to Dr. Ramesh Raskar for providing the original matlab code from which this is derived
    # Dr. Raskar's version is available here: http://web.media.mit.edu/~raskar/photo/code.pdf

    # Laplacian
    gyy = grady[1:,:-1] - grady[:-1,:-1]
    gxx = gradx[:-1,1:] - gradx[:-1,:-1]
    f = np.zeros(boundarysrc.shape)
    f[:-1,1:] += gxx
    f[1:,:-1] += gyy

    # Boundary image
    boundary = boundarysrc.copy()
    boundary[1:-1,1:-1] = 0

    # Subtract boundary contribution
    f_bp = -4*boundary[1:-1,1:-1] + boundary[1:-1,2:] + boundary[1:-1,0:-2] + boundary[2:,1:-1] + boundary[0:-2,1:-1]
    f = f[1:-1,1:-1] - f_bp

    # Discrete Sine Transform
    tt = scipy.fftpack.dst(f, norm='ortho')
    fsin = scipy.fftpack.dst(tt.T, norm='ortho').T

    # Eigenvalues
    (x,y) = np.meshgrid(range(1,f.shape[1]+1), range(1,f.shape[0]+1), copy=True)
    denom = (2*np.cos(math.pi*x/(f.shape[1]+2))-2) + (2*np.cos(math.pi*y/(f.shape[0]+2)) - 2)

    f = fsin/denom

    # Inverse Discrete Sine Transform
    tt = scipy.fftpack.idst(f, norm='ortho')
    img_tt = scipy.fftpack.idst(tt.T, norm='ortho').T

    # New center + old boundary
    result = boundary
    result[1:-1,1:-1] = img_tt

    return result
