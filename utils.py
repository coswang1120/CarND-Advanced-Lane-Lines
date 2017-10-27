# -*- coding: utf-8 -*-
"""
Lane detection related utility classes
"""
from glob import glob
from tqdm import tqdm
import cv2
import numpy as np
import os
import pickle


# distortion matrix pickle file
DISTORTION_MATRIX_FILE = 'camera_matrix.p'


class CameraCalibrator():
    """ Calibrates camera distortion using chessboard images """

    def __init__(self, image_shape, chessboard_imgs_glob, chessboard_corners):
        self.image_shape = image_shape
        self.chessboard_imgs_glob = chessboard_imgs_glob
        self.chessboard_corners = chessboard_corners

    def calc_dist_matrix(self, force=False):
        """ Calculate distortion matrix and save to file """
        if os.path.isfile(DISTORTION_MATRIX_FILE):
            with open(DISTORTION_MATRIX_FILE, 'rb') as f:
                self.camera_matrix = pickle.load(f)
        else:
            nx, ny = self.chessboard_corners
            obj_point = np.zeros((nx*ny, 3), np.float32)
            obj_point[:, :2] = np.mgrid[0:nx, 0:ny].T.reshape(-1, 2)
            obj_points = []  # list of 3D obj coordinates
            img_points = []  # list of 2D coordinates found on image plane

            # use tqdm to display progress on screen
            for img_path in tqdm(glob(self.chessboard_imgs_glob)):
                img = cv2.imread(img_path)
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)
                if ret is True:
                    obj_points.append(obj_point)
                    img_points.append(corners)

            ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
                    obj_points, img_points, self.image_shape, None, None)

            # Save the camera calibration result for later use
            self.camera_matrix = {
                'mtx': mtx,
                'dist': dist,
            }
            # Save camera matrix to avoid recalibrating every time
            # run with force=True to force recalibrate
            pickle.dump(self.camera_matrix, open(DISTORTION_MATRIX_FILE, 'wb'))

    def undistort_image(self, img):
        if not getattr(self, 'camera_matrix', None):
            self.calc_dist_matrix()

        mtx = self.camera_matrix['mtx']
        dist = self.camera_matrix['dist']
        undistorted = cv2.undistort(img, mtx, dist, None, mtx)
        return undistorted


class PerspectiveTransform():
    """ Transforms road image into a birds eye view angle """

    def __init__(self, image_shape):
        height, width = image_shape
        self.width = width
        self.height = height
        self.image_shape = image_shape

        # TODO(Olala): turn hard coded coordinates into calculated
        tls = (563, 470)  # top left source point
        bls = (220, 700)  # bottom left source point
        tld = (300, 300)  # top left destination
        bld = (300, 720)  # bottom left destination

        src = np.float32([
            [tls[0], tls[1]],
            [width - tls[0], tls[1]],
            [width - bls[0], bls[1]],
            [bls[0], bls[1]]
        ])

        dst = np.float32([
            [tld[0], tld[1]],
            [width - tld[0], tld[1]],
            [width - tld[0], bld[1]],
            [bld[0], bld[1]],
        ])

        self.transform_matrix = cv2.getPerspectiveTransform(src, dst)
        self.revert_matrix = cv2.getPerspectiveTransform(dst, src)

    def transform(self, img):
        return self._warp_perspective(img, self.transform_matrix)

    def revert(self, img):
        return self._warp_perspective(img, self.revert_matrix)

    def _warp_perspective(self, img, matrix):
        # warpPerspective expect shape in (width, height)
        shape = (self.width, self.height)
        return cv2.warpPerspective(img, matrix, shape, flags=cv2.INTER_LINEAR)


class RoadMarkingDetector():

    def __init__(self):
        self.s_magnitude_thresh = (175, 255)
        self.sobel_kernel = 7
        self.m_thresh = (14, 255)
        self.d_thresh = (0.0, 0.73)

    def find_marking_pixels(self, img):
        """ Returns a bitmap of road marking pixels """
        s_mag = self._s_magnitude(img)
        l_dir = self._l_direction(img)
        combined_binary = np.zeros_like(img[:, :, 1])
        combined_binary[(s_mag == 1) | (l_dir == 1)] = 1
        return combined_binary

    def _s_magnitude(self, img):
        """Returns magnitude thresholded binary image of
        the S channel in HLS color space
        """
        thresh = self.s_magnitude_thresh
        hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
        s_channel = hls[:, :, 2]
        magnitude_binary = np.zeros_like(s_channel)
        magnitude_binary[
            (s_channel >= thresh[0]) & (s_channel <= thresh[1])] = 1
        return magnitude_binary

    def _l_direction(self, img):
        """ Apply sobel filter on L(HLS) channel and threshold,
        filter pixels direction inside self.d_thresh and
        filter pixels magnitude inside self.m_thresh
        """
        sobel_kernel = self.sobel_kernel
        m_thresh = self.m_thresh
        d_thresh = self.d_thresh

        hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
        l_channel = hls[:, :, 1]
        sobelx = cv2.Sobel(l_channel, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
        sobely = cv2.Sobel(l_channel, cv2.CV_64F, 0, 1, ksize=sobel_kernel)

        magnitude = np.sqrt(np.square(sobelx)+np.square(sobely))
        scaled = np.uint8(255*magnitude/np.max(magnitude))
        abs_sobelx = np.absolute(sobelx)
        abs_sobely = np.absolute(sobely)
        direction = np.arctan2(abs_sobely, abs_sobelx)
        binary_output = np.zeros_like(direction)
        binary_output[
            (direction >= d_thresh[0]) &
            (direction <= d_thresh[1]) &
            (scaled >= m_thresh[0]) &
            (scaled <= m_thresh[1])] = 1
        return binary_output
