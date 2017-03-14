
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
cv2.imwrite(image_path.replace('sample','w_binary_sample'),out_w)



from image_preprocessors import combine_preprocesors
pre = combine_preprocesors(flat)
cv2.imwrite(image_path.replace('sample','pre_binary_sample'),pre)

##Finding Lines
from lane_finder import find_line
lines,curv_l,corv_r,offcenter = find_line(pre)
cv2.imwrite(image_path.replace('sample','lines_detected_sample'),lines)


from lane_finder import mix_images

mixed = mix_images(img,lines)

#
# plt.imshow(mixed)
# plt.show()


cap = cv2.VideoCapture('project_video.mp4')
# cap = cv2.VideoCapture('harder_challenge_video.mp4')
#cap = cv2.VideoCapture('challenge_video.mp4')


from collections import deque
curves = deque([])
mean_curves_count = 10


def count_mean_curve(curves):
    sum = 0
    for l,r in curves:
        sum = sum + (l+2*r)
    return  (sum/(len(curves)*2)) *0.7




size = (img.shape[1],img.shape[0])
fourcc = cv2.VideoWriter_fourcc(*'DIVX')  # 'x264' doesn't work
video = cv2.VideoWriter('001_output.avi',fourcc, 29.0, size)  # 'False' for 1-ch instead of 3-ch for color


frame_id = 0
while(cap.isOpened()):


    ret, out = cap.read()
    in_img = out
    if ret==True:



        dst = cv2.undistort(out, camear_calibration['mtx'], camear_calibration['dist'], None, camear_calibration['mtx'])
        flat = flat_perspective(dst)


        out_b = s_binnary(flat)
        cv2.imwrite(image_path.replace('sample', 'b_binary_sample'), out_b)



        out = combine_preprocesors(flat)
        out, curv_l, corv_r, offcenter = find_line(out)
        #cv2.imshow('lines', out)
        out = mix_images(in_img,out)

        curves.append((curv_l, corv_r))

        curva = (count_mean_curve(curves))
        cv2.putText(out, "Road curvature: {:.2} km".format(curva/1000), (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        sterring_angle = 1000000 / (curva ** 2) * 45
        cv2.putText(out, "Steering wheel {:02.2f} deg".format(sterring_angle), (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)


        cv2.putText(out, "Off center {:0.2f} m".format(offcenter), (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        cv2.imshow('result', out )
        if(len(curves) > mean_curves_count ):
            curves.popleft()

        filename = ("out_video/project_video/frame-{:04d}.png".format(frame_id))
        cv2.imwrite(filename,out)
        frame_id += 1

        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.imwrite("report/sample_pre.png", in_img)
            cv2.imwrite("report/sample_out.png", out)
            break
    else:
        break

cap.release()
video.release()
#cv2.destroyAllWindows()


# Walkaround for not working opencv ffmpg
# OUTPUT_DIR = 'out_video'
#
# images = sorted(glob.glob('frames/project_video/out-*.png'))
# for x in images:
#     img = cv2.imread(x)
#     dst = cv2.undistort(img, camear_calibration['mtx'], camear_calibration['dist'], None, camear_calibration['mtx'])
#     flat = flat_perspective(dst)
#
#
#     out_b = s_binnary(flat)
#     cv2.imwrite(image_path.replace('sample', 'b_binary_sample'), out_b)
#
#
#
#     out = combine_preprocesors(flat)
#     out = find_line(out)
#     out = mix_images(img,out)
#
#
#
#     filename = x.replace('frames',OUTPUT_DIR)
#     print(filename)
#     # plt.imshow(out,cmap='gray')
#     # plt.show()
#     # plt.imshow(out)
#     # plt.show()
#     cv2.imwrite(filename,out )


