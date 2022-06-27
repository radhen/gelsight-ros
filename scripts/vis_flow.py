#!/usr/bin/env python3

import numpy as np
import cv2
from gelsight import gsdevice
from gelsight_ros.util import image2markers
from find_marker import Matching

URL = "http://192.168.0.170:8080/?action=stream"
ROI = (70, 95, 335, 390)
N = 10
M = 14
FPS = 40
x0 = 13
y0 = 5
dx = 10
dy = 10

if __name__ == "__main__":
    dev = gsdevice.Camera(gsdevice.Finger.R15, URL)
    dev.connect()

    frame = None
    init_frame = None
    init_markers = None
    match = Matching(M, N, FPS, x0, y0, dx, dy)
    while dev.while_condition:
        frame = dev.get_image(ROI)
        if frame is not None:
            if init_frame is None:
                init_frame = frame

            # if init_markers is None:
            #    init_markers = image2markers(frame) 
            #    markers = init_markers
            # else:
            markers = image2markers(frame)
            
            match.init(markers)
            match.run()
            flow = match.get_flow()

            Ox, Oy, Cx, Cy, Occupied = flow
            K = 5
            for i in range(len(Ox)):
                for j in range(len(Ox[i])):
                    pt1 = (int(Ox[i][j]), int(Oy[i][j]))
                    pt2 = (int(Cx[i][j] + K * (Cx[i][j] - Ox[i][j])), int(Cy[i][j] + K * (Cy[i][j] - Oy[i][j])))
                    color = (0, 255, 255)
                    cv2.arrowedLine(frame, pt1, pt2, color, 2,  tipLength=0.2)
            
            for i in range(markers.shape[0]):
                cv2.circle(frame, (int(markers[i, 0]), int(markers[i, 1])), color=(0, 0, 0), radius=2)

            cv2.imshow("Flow Visualization", frame)
            if cv2.waitKey(1) != -1:
                break
