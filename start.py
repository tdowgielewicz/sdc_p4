from cam_calibration import calibrate_camera, flat_perspective
import pickle
import cv2
import matplotlib.pyplot as plt


REPORT = True
CALIBRATE_CAMERA = False



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
image_path = 'report/sample5.png'
img = cv2.imread(image_path)
dst = cv2.undistort(img, camear_calibration['mtx'], camear_calibration['dist'], None, camear_calibration['mtx'])
flat, TransformMatrix = flat_perspective(dst)


## Present Image Preprocesors
# Yellow
from image_preprocessors import s_binnary

out_s = s_binnary(flat)
cv2.imwrite(image_path.replace('sample','s_binary_sample'),out_s)

# B channel from LAB color spaces
from image_preprocessors import b_binnary
out_b = b_binnary(flat)
cv2.imwrite(image_path.replace('sample','b_binary_sample'),out_b)

# plt.imshow(out_b,cmap='gray')
# plt.show()

# take white channel from gray color spaces
from image_preprocessors import white_binnary
out_w = white_binnary(flat)
cv2.imwrite(image_path.replace('sample','b_binary_sample'),out_w)



from image_preprocessors import combine_preprocesors
pre = combine_preprocesors(flat)
cv2.imwrite(image_path.replace('sample','pre_binary_sample'),pre)




