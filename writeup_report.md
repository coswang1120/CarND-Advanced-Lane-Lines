##Advanced Lane Finding
###Author: HSIN-CHENG _olala7846@gmail.com_

---

**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./output_images/chessboard_original.jpg "Origin Chessboard Image"
[image2]: ./output_images/chessboard_undistort.jpg "Undistort Chessboard"
[image2.0]: ./output_images/pipeline_in.jpg "Pipeline input"
[image2.1]: ./output_images/pipeline_undist.jpg "Pipeline undistorted"
[image2.2]: ./output_images/color_stack.jpg "Pipeline Binary Stacked"
[image2.3]: ./output_images/combined_binary.jpg "Pipeline Combined Binary"

[image2.4]: output_images/test1_undistorted.jpg
[image2.5.1]: output_images/before_search.jpg 
[image2.5.2]: output_images/line_hist_search.jpg "Histogram Line Search"
[image2.5.3]: output_images/line_conv_search.jpg "Convolution Line Search"
[image2.6]: output_images/curvature_radius.jpg "Calculate radius of curvature"

[video]: ./results.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points
###Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a `writeup_resport.md` that includes all the rubric points and how I addressed each one.  
You're reading it!
###Camera Calibration

####1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in the `P4.ipynb` first code cell `LaneFinder` class `calibrate_camera` method.

I started by preparing the "object points", which will be the (x, y, z) coordinates of the chessborad corners in the world, all chessboard images under `./camera_cal/` has 9x5 corners. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

Original Image:
![alt text][image1]
Undistorted Image:
![alt text][image2]

I also store the distortion matrix in a pickle file named `distort_mtx.p` to avoid recalculate matrix every time.

###Pipeline (single images)

I implemented all pipeline steps as seperate class methods inside `LaneFinder` class and visualizes the result on `./test_images` in the second section of my `P4.jpynb` notebook.
I will only demonstrate my pipline using the `./test_images/test1.jpg` in my report, for results of other images, please reference the 2nd section of my nodebook

##### Input test1.jpg
![alt text][image2.0]

####1. Provide an example of a distortion-corrected image.
To undistort the input camera image, first we need to calibrate the camera using chessboard images, and use the calibrated matrix to undistort our input images like this.

```
lf = LaneFinder()
lf.calibarte_camera()
img = cv2.imread('./test_images/test1.jpg')
result = lf.undistort(img)
```
The implementation detail is inside the `calibrate_camera` and `undistort` method of `LaneFinder` class, The undistorted result looks like the following.
##### Undistorted image
![alt text][image2.1]


####2.1 Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.
I actually chosed to do the color transform and thresholding step **after** doing perspective transform, so I'll describe it in section 3.1 after section 3.

####3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform sits inside the `perspective_transform` method of `LaneFinder` class, it calls the `get_perspective_transform_matrix` method to generate the transform matrix. Both methods supports a keyword argument `reverse` refauts to `False`, setting `reverse=True` does the reverse perspective transform, which will be used while warping the found lane line back to the original image later.
The source and destination points in the `get_perspective_transform_matrix` method are also hand picked using `ipywidgets`.

This resulted in the following source and destination points:

|              | Source        | Destination   |
|:------------:|:-------------:|:-------------:|
| Top Left     | 563, 470      | 300, 300      |
| Right Bottom | 220, 700      | 300, 720      |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

##### Warped test1 image
![alt text][image2.4]

####3.1. Color thresholding
After perspective transform, I used a combination of color thresholds (on the S channel) and sobel gradient thresholds (on L channel) on the HLS color space to detect line pixels. The S channel thresholding did well on most cases but failed to detect correctly under massive shadows on the road, so I used the L channel gradient magnitude and direction thresholding to complement on that part. I hand picked the parameters using `ipywidgets` library.
Please see the `combined_thresholding`, `l_direction` and `s_magnitude` methods for more implementaiton detail. 
The result looks like this.
##### Stacked binary 
(red: L channel sobel thresholding, green: S channel magnitude thresholding)
![alt text][image2.2]
##### Combined binary result
![alt text][image2.3]
For more stacked images, please see the second section of my [notebook](./P4.ipynb)


####4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

For finding the lane lines, I first use a histogram search to find lane lines naively, once I got an approximately position about the lines, I use a convolutional search to find a more accurate position about the lines. The implementaion detail is inside the `find_lines`, `histogram_find_lines` and `convolution_find_lines` methods of the **LaneFinder** class

I verified this by visualizing the results:
#####Original image
![alt text][image2.5.1]
#####Histogram search
![alt text][image2.5.2]
#####Convolutional search
![alt text][image2.5.3]
Again, more visualiztion samples are under the 2nd section of my [notebook](./P4.ipynb)

####5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I calculate the radius of the curvature by calculating the second dirivitive of the fitted polynomial over the bottom of the fitted line, and I assumed the warped pixel space has `30/720` meters per pixel height and `3.7/700` meters per pixel width. (please see the `curvature_radius` method in the **Line** class for more implementaion detail).
As the offset to the center, I calculate the x pixel at the bottom of both lines and average them as the center of the lines, and convert the pixel difference between center of the lines and center of the image into meters.
See `img_pipeline` for more detail
The result will be demonstrated in the next section.

####6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I plotted back the road section using the same warping function as doing perspective transform but with keyword parameter `inveres=True`. And visualized it on the origin undistorted image using `cv2.addWeighted(undist, 1.0, road, 0.3, 0)`
please reference `img_pipeline` for more detail

![alt text][image2.6]

---

###Pipeline (video)

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's my result  [video] or [youtube link](https://youtu.be/MIyqcMWk0jo)


---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

First I implemented all steps as the course material and use the `ipywidgets.interact` to find the best parameters for thresholding and perspective transform on jupyter notebook, but I soon find out that the thresholded lines become thicker or even two seperate liens after perspective transform, so I decided to move the perspective transform step forward before the thresholding step. This helps me tune the thresholding parameters more focus on only the road and lanes.

After that, my detection pipe line works well on 90% of the frames the project video, but failed to detect correclty when at around 23 second and 40 seconds, and I found out that's because 

1. The shadow of the tree cause the S channel fails to detect the lines correctly.
2. The bumpy road causes camera to tilt front and back, which makes the perspective transform matrix changes.

I solved these problems by recording the fitted lines of last 5 frames, and added a sanity check before appending any results into it. I used a python `deque` with max length 5 to record the latest 5 fitted lines. By averaging the coefficient of the past 5 lines and skipping bad frames, My result video looks smoother than before.
Reference `line_valid` in **Line** class for the sanity check logic.
