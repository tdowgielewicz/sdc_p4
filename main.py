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

    # fig = plt.figure()
    # ax = fig.add_subplot(2,1,1)
    # ax.imshow(img)
    # ax2 = fig.add_subplot(2, 1, 2)
    # ax2.imshow(out)
    # ax2.autoscale(False)
    # plt.show()
    return out


S_threshold = (80, 255)
H_threshold = (1, 120)

def s_treshording(img):

    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    # H = hls[:, :, 0]
    # L = hls[:, :, 1]
    S = hls[:, :, 2]

    H = hls[:, :, 0]

    thresh = H_threshold
    h_mask = np.zeros_like(H)
    h_mask[(H > thresh[0]) & (H <= thresh[1])] = 1


    thresh = S_threshold
    binary = np.zeros_like(S)
    binary[(S > thresh[0]) & (S <= thresh[1])] = 255
    binary = binary*h_mask
    #binary = cv2.cvtColor(binary, cv2.COLOR_BGR2GRAY)
    # plt.imshow(binary)
    # plt.show()
    # fig = plt.figure()
    # ax = fig.add_subplot(1,2,1)
    # ax.imshow(img)
    # ax2 = fig.add_subplot(1, 2, 2)
    # ax2.imshow(binary,cmap='Greys_r')
    # ax2.autoscale(False)
    # plt.show()

    return binary

def h_treshording(img):

    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    H = hls[:, :, 0]

    thresh = H_threshold
    binary = np.zeros_like(H)
    binary[(H > thresh[0]) & (H <= thresh[1])] = 255

    # fig = plt.figure()
    # ax = fig.add_subplot(1,2,1)
    # ax.imshow(img)
    # ax2 = fig.add_subplot(1, 2, 2)
    # ax2.imshow(binary,cmap='Greys_r')
    # ax2.autoscale(False)
    # plt.show()

    return binary

def dir_threshold(img, sobel_kernel=3, thresh=(0, np.pi/2)):
    # Grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Calculate the x and y gradients
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # Take the absolute value of the gradient direction,
    # apply a threshold, and create a binary image result
    absgraddir = np.arctan2(np.absolute(sobely), np.absolute(sobelx))
    binary_output =  np.zeros_like(absgraddir)
    binary_output[(absgraddir >= thresh[0]) & (absgraddir <= thresh[1])] = 1

    # Return the binary image
    return binary_output



#report flatening images
if REPORT:
    img = cv2.imread('report/sample3.png')
    flat = flat_perspective(img)
    cv2.imwrite("report/flat_sample3.png", flat)


if REPORT:
    id = '_pre'
    img = cv2.imread('report/sample{}.png'.format(id))
    out = s_treshording(img)
    cv2.imwrite("report/s_thershold{}.png".format(id), out)
    out = h_treshording(img)
    cv2.imwrite("report/h_thershold{}.png".format(id), out)


    # images = glob.glob('report/sample*.png')
    # for image_path in images:
    #     img = cv2.imread(image_path)
    #     out = s_treshording(img)
    #     new_path = image_path.replace('sample','s_treshold_sample_')
    #     cv2.imwrite(new_path, out)


def image_preprocesor(img):
    #1 Fix camera disortion
    img = cv2.undistort(img, camear_calibration['mtx'], camear_calibration['dist'], None, camear_calibration['mtx'])
    #out = s_treshording(img)
    out = dir_threshold(img, sobel_kernel=5, thresh=(0.7, 1.3))
    kernel = np.ones((2, 3), np.uint8)
    out = cv2.erode(out, kernel, iterations=1)


    return out

# if True:
#     exit()

cap = cv2.VideoCapture('project_video.mp4')

while(cap.isOpened()):

    ret, frame = cap.read()
    if ret==True:


        # write the flipped frame
        #out.write(frame)

        #out = s_treshording(frame)
        # out = dir_threshold(img)
        out = image_preprocesor(frame)
        out =flat_perspective(out)


        cv2.imshow('frame',out)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.imwrite("report/sample_pre.png", frame)
            cv2.imwrite("report/sample_out.png", out)
            break
    else:
        break

cap.release()
cv2.destroyAllWindows()







