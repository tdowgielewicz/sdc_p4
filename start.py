
import glob
import pickle
import cv2
import matplotlib.pyplot as plt
import numpy as np


REPORT = True
CALIBRATE_CAMERA = False


from cam_calibration import calibrate_camera, flat_perspective

if CALIBRATE_CAMERA:
    camear_calibration = calibrate_camera()
else:
    camear_calibration = pickle.load(open( "camera_calibration.p", "rb" ) )


#Check Calibration
img = cv2.imread('report/sample3.png')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imwrite("report/before_calibration_road.png",img)
dst = cv2.undistort(img, camear_calibration['mtx'], camear_calibration['dist'], None, camear_calibration['mtx'])
cv2.imwrite("report/after_calibration_road.png", dst)


#flatten perspective
image_path = 'report/sample1.png'
img = cv2.imread(image_path)
dst = cv2.undistort(img, camear_calibration['mtx'], camear_calibration['dist'], None, camear_calibration['mtx'])
flat = flat_perspective(dst)


## Present Image Preprocesors
# Yellow
from image_preprocessors import s_binnary

out_s = s_binnary(flat)
cv2.imwrite(image_path.replace('sample','s_binary_sample'),out_s)
# plt.imshow(out_s,cmap='gray')
# plt.show()


# B channel from LAB color spaces
from image_preprocessors import b_binnary
out_b = b_binnary(flat)
cv2.imwrite(image_path.replace('sample','b_binary_sample'),out_b)



# take white channel from gray color spaces
from image_preprocessors import white_binnary
out_w = white_binnary(flat)
cv2.imwrite(image_path.replace('sample','b_binary_sample'),out_w)



from image_preprocessors import combine_preprocesors
pre = combine_preprocesors(flat)
cv2.imwrite(image_path.replace('sample','pre_binary_sample'),pre)

##Finding Lines
from lane_finder import find_line
lines = find_line(pre)


from lane_finder import mix_images

mixed = mix_images(img,lines)


plt.imshow(mixed)
plt.show()

OUTPUT_DIR = 'out_video'

images = sorted(glob.glob('frames/project_video/out-*.png'))
for x in images:
    img = cv2.imread(x)
    dst = cv2.undistort(img, camear_calibration['mtx'], camear_calibration['dist'], None, camear_calibration['mtx'])
    flat = flat_perspective(dst)


    out_b = s_binnary(flat)
    cv2.imwrite(image_path.replace('sample', 'b_binary_sample'), out_b)



    out = combine_preprocesors(flat)
    out = find_line(out)
    out = mix_images(img,out)



    filename = x.replace('frames',OUTPUT_DIR)
    print(filename)
    # plt.imshow(out,cmap='gray')
    # plt.show()
    # plt.imshow(out)
    # plt.show()
    cv2.imwrite(filename,out )


