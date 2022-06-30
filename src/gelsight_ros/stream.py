#!/usr/bin/env python3

from re import I
import cv2 as cv
from gelsight import gsdevice
import numpy as np

class GelsightStream:
    def while_condition(self) -> bool:
        raise NotImplementedError()

    def get_frame(self) -> np.ndarray:
        raise NotImplementedError()

class GelsightHTTPStream(GelsightStream):
    def __init__(self, url: str, roi: tuple[int, int, int, int]):
        super().__init__() 
        self._roi = roi

        self._dev = gsdevice.Camera(gsdevice.Finger.R15, url)
        self._dev.connect()

    def while_condition(self) -> bool:
        return self._dev.while_condition        

    def get_frame(self) -> np.ndarray:
        """Returns cropped image frame from GelSight in RGB format"""
        return self._dev.get_image(self._roi)

class GelsightFileStream(GelsightStream):
    def __init__(self, path: str):
        super().__init__()
        self._roi = roi

        self._vid = cv2.VideoCapture(path)

    def while_condition(self) -> bool:
        return self._vid.isOpened()

    def get_frame(self) -> np.ndarray:
        ret, frame = self._vid.read()
        if not ret:
            raise RuntimeError("Failed to collect frame from video feed.")
        return frame