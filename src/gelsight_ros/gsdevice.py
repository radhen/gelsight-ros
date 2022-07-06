import cv2
import numpy as np
import enum

# creating enumerations using class
class Finger(enum.Enum):
    R1 = 1
    R15 = 2
    DIGIT = 3


def warp_perspective(img, corners, output_sz):
    TOPLEFT, TOPRIGHT, BOTTOMRIGHT, BOTTOMLEFT = corners
    WARP_H = output_sz[0]
    WARP_W = output_sz[1]
    points1 = np.float32([TOPLEFT, TOPRIGHT, BOTTOMRIGHT, BOTTOMLEFT])
    points2 = np.float32([[0, 0], [WARP_W, 0], [WARP_W, WARP_H], [0, WARP_H]])
    matrix = cv2.getPerspectiveTransform(points1, points2)
    result = cv2.warpPerspective(img, matrix, (WARP_W, WARP_H))
    return result

class Camera:
    def __init__(self, dev_type, dev_id):
        # variable to store data
        self.data = None
        self.name = dev_type
        self.dev_id = dev_id
        self.imgw = 320 # this is for R1, R1.5 is 240
        self.imgh = 240 # this is for R1, R1.5 is 320
        self.cam = None
        self.while_condition = 1

    def connect(self):

        # This cam uses the ximea api, xiapi
        if self.name == Finger.R1:

            from ximea import xiapi

            self.cam = xiapi.Camera(self.dev_id)
            self.cam.open_device()

            # Set the ximea camera parameters
            self.cam.set_gpo_selector('XI_GPO_PORT1')
            self.cam.set_acq_timing_mode('XI_ACQ_TIMING_MODE_FRAME_RATE_LIMIT')
            self.cam.set_gpi_mode('XI_GPI_OFF')
            self.cam.set_imgdataformat('XI_RGB24')
            self.cam.set_wb_kr(1.0)
            self.cam.set_wb_kg(1.0)
            self.cam.set_wb_kb(1.0)
            # self.cam.set_gammaC(0.5)
            self.cam.set_framerate(60)
            self.cam.set_gpo_mode('XI_GPO_EXPOSURE_ACTIVE')
            self.cam.set_exposure(14000)  ## microseconds
            self.cam.set_manual_wb(1)
            self.cam.start_acquisition()

        # This camera is the pi camera and uses open cv to get the video data from the streamed video
        elif self.name == Finger.R15:
            self.cam = cv2.VideoCapture(self.dev_id)
            if self.cam is None or not self.cam.isOpened():
                print('Warning: unable to open video source: ', self.dev_id)
            self.imgw = 120 #int(self.cam.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.imgh = 160 #int(self.cam.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.fps = int(self.cam.get(cv2.CAP_PROP_FPS))
            print('R1.5 image size = %d x %d at %d fps', self.imgw, self.imgh, self.fps)

        # not supporting this yet
        elif self.name == Finger.DIGIT:
            print('Digit is not supported yet')

        # any others ???
        else:
            print('Unknown device type ', self.name)
            print('Please select one of Finger', list(map(lambda d: d.name, Finger)))

        if self.name == Finger.R15:
            self.while_condition = self.cam.isOpened()

        return self.cam

    def get_raw_image(self):
        # Use ximea api, xiapi
        if self.name == Finger.R1:
            from ximea import xiapi
            # create image handle
            self.img = xiapi.Image()
            self.cam.get_image(self.img)
            f0 = self.img.get_image_data_numpy()
            f0 = cv2.resize(f0, (self.imgw, self.imgh))
        else:
            ret, f0 = self.cam.read()
        self.data = cv2.cvtColor(f0, cv2.COLOR_BGR2RGB)
        return self.data


    def get_image(self, roi):
        # Use ximea api, xiapi
        if self.name == Finger.R1:
            from ximea import xiapi
            # create image handle
            self.img = xiapi.Image()
            self.cam.get_image(self.img)
            f0 = self.img.get_image_data_numpy()
            f0 = cv2.resize(f0, (self.imgw, self.imgh))
        else:
            # read the raw image
            ret, f0 = self.cam.read()
            # set warping parameters depending on the streamed image size
            # currently only supports streaming 640x480 or 320x240
            # see mjpg_streamer command running on the raspberry pi
            if f0.shape == (640, 480, 3):
                xorigin = 25
                yorigin = 25
                dx = 35
            elif f0.shape == (320, 240, 3):
                xorigin = 12
                yorigin = 12
                dx = 18

            # crop the raw image
            f0 = f0[int(roi[1]):int(roi[1]+roi[3]), int(roi[0]):int(roi[0]+roi[2])]
            xdim = f0.shape[1]
            ydim = f0.shape[0]

            # warp the trapezoid, top left start, clockwise
            f0 = warp_perspective(f0,
                                [[xorigin, yorigin], [xdim-xorigin, yorigin],
                                [xdim-xorigin-dx, ydim-yorigin], [xorigin+dx, ydim-yorigin]],
                                output_sz=(self.imgh, self.imgw))

        self.data = cv2.cvtColor(f0, cv2.COLOR_BGR2RGB)
        return self.data

    def save_image(self, fname):
         cv2.imwrite(fname, self.data)

    def start_video(self):
        # the default while condition is set to 1, change it for R1.5
        while( self.while_condition ):
            frame = self.get_image()
            cv2.imshow('frame', frame)
            cv2.waitKey(0)

    def stop_video(self):
        # use the ximea api xiapi for R1
        if self.name == Finger.R1:
            self.cam.stop_acquisition()
            self.cam.close_device()
        cv2.destroyAllWindows()




