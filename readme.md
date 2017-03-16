## Self Driving Car - Advanced Lane Line Detection

---

**Project Scope**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.




[image2]: ./test_images/test1.jpg "Road Transformed"
[image3]: ./examples/binary_combo_example.jpg "Binary Example"
[image4]: ./examples/warped_straight_lines.jpg "Warp Example"
[image5]: ./examples/color_fit_lines.jpg "Fit Visual"
[image6]: ./examples/example_output.jpg "Output"
[video1]: ./project_video.mp4 "Video"

###


---

#### This file
 This file README.md includes all the rubric points. list can be found [here](https://review.udacity.com/#!/rubrics/571/view)

#### Camera Calibration
Before Calibration:

<img src="https://github.com/muncz/sdc_p4/blob/master/report/before_calibration.png" width="400">

After Calibration:

<img src="https://github.com/muncz/sdc_p4/blob/master/report/after_calibration.png" width="400">


##### Camera disortion correction


The code for this step is contained in the function  calbrate_camera() in [cam_calibaration module](cam_calibration.py)

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

On road before calibration:

<img src="https://github.com/muncz/sdc_p4/blob/master/report/before_calibration_road.png" width="400">

On road after calibration:

<img src="https://github.com/muncz/sdc_p4/blob/master/report/after_calibration_road.png" width="400">

There is only slighty difference from before to after images. But this small diference can do a real job when measuring road curvature

### Pipeline


#### 1.Camera undisortion

Explained few lines above

#### 2. Birdeye view

To save processing power and to not aplly threshold, edge detection etc to whole images. I  mannaged to do perespective transform on the beginning

| Source        | Destination   |
|:-------------:|:-------------:|
| 585, 460      | 80, 80        |
| 710, 460      | 520, 10       |
| 1000, 623     | 520, 790      |
| 373, 623      | 80, 790       |

<img src="https://github.com/muncz/sdc_p4/blob/master/report/flatten_points_map.png" width="400">

The code for this step is contained in the function  flat_perspective(img) in [cam_calibaration module](cam_calibration.py)

Result of perspective transform:

 <img src="https://github.com/muncz/sdc_p4/blob/master/report/flat_sample2.png" width="200">

Perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.


#### 3. Images color exctraction

##### S Channel
    S_channel = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)[:,:,2]
    threshold= [150,255]


<img src="https://github.com/muncz/sdc_p4/blob/master/report/s_binary_sample2.png" width="200">

##### B chaneel
    B_channel = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)[:,:,0]
    threshold= [155,255]

<img src="https://github.com/muncz/sdc_p4/blob/master/report/b_binary_sample3.png" width="200">

##### White color

    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    threshold= [225,255]

 <img src="https://github.com/muncz/sdc_p4/blob/master/report/w_binary_sample2.png" width="200">



##### Logic summary

My images is logic opperation of ( S_Channel * B_Channel )  + White
The code for this step is contained in the function combine_preprocesors(img) in [Preprocessor module](image_preprocessors.py)


 <img src="https://github.com/muncz/sdc_p4/blob/master/report/pre_binary_sample2.png" width="200">


### Lines description

Curvature and lane lines pixel identification can be found in function find_line(img) in [Lane finder module ](lane_finder.py)
#### Lane Lines pixel identification


In order to detect lane lines, first histogram was used to have good start point i used histogram of bootom half of image. Than divided this in a half on x axis peaks were the lines.
Than i made two crawling windows (for left and right line) that was moving upwards to find next pixels. If some pixels were found than windows were repositioned to the center.

 <img src="https://github.com/muncz/sdc_p4/blob/master/report/lines_detected_sample1c.png" width="200">

That same time each pixel that was inside that window was save as point to find line:

        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (
        nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (
        nonzerox < win_xright_high)).nonzero()[0]
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)

Next, to idientify line Numpy function (fitpoly) was used, this is looking for a polynomial of 2nd order that best matches our found points.
        left_fit = np.polyfit(lefty, leftx, 2)
        right_fit = np.polyfit(righty, rightx, 2)

 <img src="https://github.com/muncz/sdc_p4/blob/master/report/lines2.png" width="200">


#### Curvature identification

Already having the points of the circle we can try to fit it in to radius of curvature function. [Link](http://www.intmath.com/applications-differentiation/8-radius-curvature.php)

        left_fit_cr = np.polyfit(lefty * ym_per_pix, leftx * xm_per_pix, 2)
        right_fit_cr = np.polyfit(righty * ym_per_pix, rightx * xm_per_pix, 2)

        left_curverad = ((1 + (2 * left_fit_cr[0] * y_eval * ym_per_pix + left_fit_cr[1]) ** 2) ** 1.5) / np.absolute(
            2 * left_fit_cr[0])
        right_curverad = (
                         (1 + (2 * right_fit_cr[0] * y_eval * ym_per_pix + right_fit_cr[1]) ** 2) ** 1.5) / np.absolute(
            2 * right_fit_cr[0])

Please notice variables ym_per_pix and xm_per_pix which scales pixels world to real S.I. values world

        ym_per_pix = 30 / 720  # meters per pixel in y dimension
        xm_per_pix = 3.7 / 700  # meters per pixel in x dimension


However after all this actions. Curvature was imho to big (should be around 1km on the most part of the track, so at the very end it was manualy adjusted by multiplying by factor - 0.7


---

### Video Pipline

Get frame -> Remove Disortion -> Transform Perspective -> Image Preprocessors -> Find Lines -> Draw Lines ->  Retransform Perspective -> Mix with orginal frame -> Put some text about

Here's a link to my [video](result.mp4) result in this repo repo.


---

### Summary

This pipe line works good on the simple videos with quite constant lighting conditions. However a lot of thing could be improved. A lot of adjustments still should be done to have 99.9999% robust system
This pipleine for shure will fail in diferent light conditions it is adjusted to that only one video. Also finding the lines works good only on quite small Curvatures. Also processing power is the limit,
I have quite buffy PC and i was not able to process image in real time. However there may some solutions like using GPU, switch to C++ and limit ROIs with preprocessing
Nevertheless it is possible to make more robust system based on this one.
