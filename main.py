import pickle
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob


REPORT = True
CALIBRATE_CAMERA = False

def calibrate_camera():
    nx = 9
    ny = 6
    images = glob.glob('camera_calibration/calibration*.jpg')


    objpoints = []
    imgpoints = []

    objp = np.zeros((nx * ny, 3), np.float32)
    objp[:, :2] = np.mgrid[0:nx, 0:ny].T.reshape(-1, 2)


    for image_path in images:
        img = cv2.imread(image_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)

        if ret:
            imgpoints.append(corners)
            objpoints.append(objp)
            #img = cv2.drawChessboardCorners(img, (nx, ny), corners, ret)

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

    camear_calibration = {'dist':dist, 'mtx':mtx}
    pickle.dump(camear_calibration, open("camera_calibration.p", "wb"))
    return camear_calibration

if CALIBRATE_CAMERA:
    camear_calibration = calibrate_camera()
else:
    camear_calibration = pickle.load(open( "camera_calibration.p", "rb" ) )



#save images for report purpose
if REPORT:
    img = cv2.imread('report/sample2.png')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imwrite("report/before_calibration_road.png",img)
    dst = cv2.undistort(img, camear_calibration['mtx'], camear_calibration['dist'], None, camear_calibration['mtx'])
    cv2.imwrite("report/after_calibration_road.png", dst)



def flat_perspective(img):
    #points taken manualy in paint
    img_size = (400, 800)
    offset = {'x':80,'y':10}
    p1 = [610,440]
    p4 = [373,623]
    p3 = [1000,623]
    p2 = [680,444]

    src = np.float32([p1,p2,p3,p4])
    dst = np.float32([[offset['x'], offset['x']], [img_size[0] - offset['x'], offset['y']],
                      [img_size[0] - offset['x'], img_size[1] - offset['y']],
                      [offset['x'], img_size[1] - offset['y']]])
    M = cv2.getPerspectiveTransform(src, dst)
    out = cv2.warpPerspective(img, M, img_size)
    # plt.imshow(out)
    # plt.show()
    return out


def s_treshording(img):

    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    # H = hls[:, :, 0]
    # L = hls[:, :, 1]
    S = hls[:, :, 2]

    thresh = (90, 255)
    binary = np.zeros_like(S)
    binary[(S > thresh[0]) & (S <= thresh[1])] = 1
    plt.imshow(binary)
    plt.show()

    return binary

def h_treshording(img):

    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    H = hls[:, :, 0]

    thresh = (90, 255)
    binary = np.zeros_like(H)
    binary[(H > thresh[0]) & (H <= thresh[1])] = 1
    # plt.imshow(out)
    # plt.show()

    return binary


def image_preprocesor(img):
    #1 Fix camera disortion
    img = cv2.undistort(img, camear_calibration['mtx'], camear_calibration['dist'], None, camear_calibration['mtx'])

    return img

#report flatening images
if REPORT:
    img = cv2.imread('report/sample3.png')
    flat = flat_perspective(img)
    cv2.imwrite("report/flat_sample3.png", flat)


if REPORT:
    images = glob.glob('report/sample*.png')
    for image_path in images:
        img = cv2.imread(image_path)
        out = s_treshording(img)
        new_path = image_path.replace('sample','s_treshold_sample_')
        cv2.imwrite(new_path, out)



#cap = cv2.VideoCapture('project_video.mp4')
#
# while(cap.isOpened()):
#
#     ret, frame = cap.read()
#     if ret==True:
#
#
#         # write the flipped frame
#         #out.write(frame)
#
#         cv2.imshow('frame',frame)
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             cv2.imwrite("report/sample.png", frame)
#             break
#     else:
#         break
#
# cap.release()
# cv2.destroyAllWindows()







