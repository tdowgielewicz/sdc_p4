import cv2
import numpy as np
import matplotlib.pyplot as plt

from cam_calibration import get_perspective_points


n_windows = 12
slider_width = 150
minpix = 20
margin = 50


#havly inspired by lesson example
def find_line(img):
    histogram = np.sum(img[int(img.shape[0] / 2):,:], axis=0)
    out_img = np.dstack((img,img,img))

    nonzero = img.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []


    image_heigth = img.shape[1]
    image_width = img.shape[0]
    center = int(image_width/2)

    slider_heigth = int(image_heigth/n_windows)+15


    leftx_base = np.argmax(histogram[:center])
    rightx_base = np.argmax(histogram[center:]) + center

    # Current positions to be updated for each window
    leftx_current = leftx_base
    rightx_current = rightx_base

    for window in range(n_windows):

        win_y_low = img.shape[0] - (window + 1) * slider_heigth
        win_y_high = img.shape[0] - window * slider_heigth
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin

        cv2.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),(0,255,0), 2)
        cv2.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),(0,255,0), 2)

        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (
        nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (
        nonzerox < win_xright_high)).nonzero()[0]
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    # Concatenate the arrays of indices
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    # Fit a second order polynomial to each
    try:
        left_fit = np.polyfit(lefty, leftx, 2)
        right_fit = np.polyfit(righty, rightx, 2)

        ploty = np.linspace(0, img.shape[0] - 1, img.shape[0])
        left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
        right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

        out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
        out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]
        # plt.imshow(out_img)
        # plt.plot(left_fitx, ploty, color='yellow')
        # plt.plot(right_fitx, ploty, color='yellow')
        # plt.xlim(0, image_heigth)
        # plt.ylim(image_width, 0)
    except:
        pass
    # plt.show()
    return out_img

def mix_images(orginal,wraped):
    src, dst, img_size = get_perspective_points()
    Minv = cv2.getPerspectiveTransform(dst,src)

    warp_zero = np.zeros_like(wraped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))
    newwarp = cv2.warpPerspective(wraped, Minv, (orginal.shape[1], orginal.shape[0]))

    result = cv2.addWeighted(orginal, 1, newwarp, 0.7, 0)

    return result












