import cv2
import numpy as np
import matplotlib.pyplot as plt

from cam_calibration import get_perspective_points


n_windows = 12
slider_width = 150
minpix = 20
margin = 100


HISTORY_RIGHT = []
HISTORY_LEFT = []

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

    lines_center = (rightx_base + leftx_base ) /2

    CAMERA_CENTER_OFFSET = -100

    xm_per_pix = 3.7 / 700  # meters per pixel in x dimension

    lines_offset = (center - lines_center + CAMERA_CENTER_OFFSET) * xm_per_pix
    #print(lines_offset)

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

        cv2.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),(0,0,255), 5)
        cv2.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),(255,0,0), 5)

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
    #
    #print(lefty,leftx)
    try:
        left_fit = np.polyfit(lefty, leftx, 2)
        right_fit = np.polyfit(righty, rightx, 2)


        if left_fit[0] <0 :
            isLeft = -1
        else:
            isLeft = 1

        ploty = np.linspace(0, img.shape[0] - 1, img.shape[0])
        left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
        right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

        out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [0, 255, 0]
        out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 255, 0]

        # plt.imshow(out_img)
        # plt.plot(left_fitx, ploty, color='yellow')
        # plt.plot(right_fitx, ploty, color='yellow')
        # plt.xlim(0, image_heigth)
        # plt.ylim(image_width, 0)
        # plt.show()
        # plt.savefig('report/lines.jpg')


        y_eval = np.max(ploty)
        # left_curverad = ((1 + (2 * left_fit[0] * y_eval + left_fit[1]) ** 2) ** 1.5) / np.absolute(2 * left_fit[0])
        # right_curverad = ((1 + (2 * right_fit[0] * y_eval + right_fit[1]) ** 2) ** 1.5) / np.absolute(2 * right_fit[0])
        #print(left_curverad, right_curverad)

        ym_per_pix = 40 / 800  # meters per pixel in y dimension
        xm_per_pix = 4 / 400  # meters per pixel in x dimension

        #Fit new polynomials to x,y in world space
        left_fit_cr = np.polyfit(lefty * ym_per_pix, leftx * xm_per_pix, 2)
        right_fit_cr = np.polyfit(righty * ym_per_pix, rightx * xm_per_pix, 2)
        # Calculate the new radii of curvature
        left_curverad = ((1 + (2 * left_fit_cr[0] * y_eval * ym_per_pix + left_fit_cr[1]) ** 2) ** 1.5) / np.absolute(
            2 * left_fit_cr[0])
        right_curverad = (
                         (1 + (2 * right_fit_cr[0] * y_eval * ym_per_pix + right_fit_cr[1]) ** 2) ** 1.5) / np.absolute(
            2 * right_fit_cr[0])
        # Now our radius of curvature is in meters
        # print(left_curverad, 'm', right_curverad, 'm')
        # Recast the x and y points into usable format for cv2.fillPoly()


        pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])

        global HISTORY_LEFT, HISTORY_RIGHT
        # print("pts",pts_left.sum(),pts_left.sum())
        # print("crv:", left_curverad,right_curverad)

        R = 0
        B = 0
        G = 255

        #check if left and right curvature match
        is_curvature_ok = True
        if( right_curverad < left_curverad * 0.3 ) or ( right_curverad  > left_curverad * 2):
            is_curvature_ok  = False
            # R =  255
            # G = 0

        if (not is_curvature_ok):
            pts_left = HISTORY_LEFT[-1]
            pts_right = HISTORY_RIGHT[-1]


        if (pts_left.sum() < 315000 ):
            pts_left = HISTORY_LEFT[-1]
            # R+= 200
            # G -=50
        else:
            HISTORY_LEFT.append(pts_left)

        if (pts_right.sum() < 315000 ):
            pts_right = HISTORY_RIGHT[-1]
            # B += 200
            # G -= 50
        else:
            HISTORY_RIGHT.append(pts_right)

        pts = np.hstack((pts_left, pts_right))

        cv2.fillPoly(out_img, np.int_([pts]), (R, G, B))


        return out_img, left_curverad, right_curverad, lines_offset, isLeft

    except Exception as e:
        print("COULD NOT FIT CIRCLE, Using last one",e)

        pts = np.hstack((HISTORY_LEFT[-2], HISTORY_RIGHT[-2]))
        print("left:",HISTORY_LEFT[-2])
        print("right:",HISTORY_RIGHT[-2])
        # plt.imshow(out_img)
        # plt.show()
        cv2.fillPoly(out_img, np.int_([pts]), (0, 255, 0))
        #
        # plt.imshow(out_img)
        # plt.show()
        return out_img, -1, -1, 0, False



    #plt.show()








def mix_images(orginal,wraped):
    src, dst, img_size = get_perspective_points()
    Minv = cv2.getPerspectiveTransform(dst,src)

    warp_zero = np.zeros_like(wraped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    newwarp = cv2.warpPerspective(wraped, Minv, (orginal.shape[1], orginal.shape[0]))
    # plt.imshow(orginal)
    # plt.show()
    # print(warp_zero.shape,orginal.shape)
    result = cv2.addWeighted( newwarp, 0.3,orginal, 1, 0.1)



    return result












