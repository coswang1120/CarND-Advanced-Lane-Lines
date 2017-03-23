# -*- coding: utf-8 -*-

from moviepy.editor import VideoFileClip
import cv2
import numpy as np
from lanelines import LaneFinder

lf = LaneFinder()


def img_pipeline(img, visualize=False):

    undist = lf.undistort(img)
    warped = lf.perspective_transform(undist)
    combined_binary = lf.combined_thresholding(warped)
    left_fit, right_fit = lf.find_lines(combined_binary)
    result = np.dstack((combined_binary, combined_binary, combined_binary))*255

    ploty = np.linspace(0, lf.height-1, lf.height)
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array(
        [np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    cp = np.zeros_like(result)
    cv2.fillPoly(cp, np.int_([pts]), (0, 255, 0))
    road = lf.perspective_transform(cp, reverse=True)
    result = cv2.addWeighted(undist, 1.0, road, 0.3, 0)

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


# NOTE: f1_image function expects color images!!
outfile = 'extracted_results.mp4'
clip1 = VideoFileClip("challenge_video.mp4").subclip(0, 5)
white_clip = clip1.fl_image(img_pipeline)
white_clip.write_videofile(outfile, audio=False)
