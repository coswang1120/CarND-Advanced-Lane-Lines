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

    def __init__(self, image_size, chessboard_imgs_glob, chessboard_corners):
        self.image_size = image_size
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
                    obj_points, img_points, self.image_size, None, None)

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
