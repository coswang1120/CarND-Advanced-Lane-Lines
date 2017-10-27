# -*- coding: utf-8 -*-
from lanelines import LaneFinder
from moviepy.editor import VideoFileClip
from utils import CameraCalibrator
from utils import PerspectiveTransform
from utils import RoadMarkingDetector
import cv2
import numpy as np

lf = LaneFinder()
INPUT_FILE_NAME = 'project_video.mp4'
OUTPUT_FILE_NAME = 'project_result.mp4'
IMAGE_SHAPE = (720, 1280)
CHESSBOARD_IMGS_GLOB = './camera_cal/*.jpg'
CHESSBOARD_CORNERS = (9, 5)

camera_calibrator = CameraCalibrator(
    IMAGE_SHAPE, CHESSBOARD_IMGS_GLOB, CHESSBOARD_CORNERS)
p_transformer = PerspectiveTransform(IMAGE_SHAPE)
marking_detector = RoadMarkingDetector()


def lane_detection(img, visualize=False):

    img = camera_calibrator.undistort_image(img)

    warped = p_transformer.transform(img)
    combined_binary = marking_detector.find_marking_pixels(warped)
    left_fit, right_fit = lf.find_lines(combined_binary)
    result = np.dstack((combined_binary, combined_binary, combined_binary))*255
    return result

    ploty = np.linspace(0, lf.height-1, lf.height)
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array(
        [np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    cp = np.zeros_like(result)
    cv2.fillPoly(cp, np.int_([pts]), (0, 255, 0))
    road = p_transformer.revert(cp)
    result = cv2.addWeighted(img, 1.0, road, 0.3, 0)

    # Calculate curvature radius of both lines and average them
    left_rad = lf.left_lines.current_curvature_radius()
    right_rad = lf.right_lines.current_curvature_radius()
    mean_rad = (left_rad + right_rad)/2

    # Calculate center of the road to the center of the image
    left_x = lf.left_lines.best_x()   # left line bottom pixel x position
    right_x = lf.right_lines.best_x()  # right line bottom pixel x position
    offset_x = (1280/2) - (left_x + right_x)/2
    offset_direction = "right" if offset_x > 0 else "left"
    offset_x_meter = offset_x * 3.7/700

    # write radius and offset onto image
    result = cv2.putText(
        result, 'Radius of Curvature = %.0f' % (mean_rad),
        (50, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
    result = cv2.putText(
        result, 'Vehicle is %.2f m %s of center' % (
            abs(offset_x_meter), offset_direction),
        (50, 140), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
    return result


if __name__ == "__main__":
    camera_calibrator.calc_dist_matrix()
    clip1 = VideoFileClip(INPUT_FILE_NAME).subclip(0, 5)
    white_clip = clip1.fl_image(lane_detection)
    white_clip.write_videofile(OUTPUT_FILE_NAME, audio=False)
