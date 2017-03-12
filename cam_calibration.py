import glob
import numpy as np
import cv2
import pickle

def calibrate_camera():
    print("started calibrating camera...")
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
    filename = "camera_calibration.p"
    pickle.dump(camear_calibration, open(filename, "wb"))
    print("Camera calibration file created: ",filename)
    return camear_calibration



def flat_perspective(img):
    #points taken manualy in paint
    img_size = (600, 800)
    offset = {'x':80,'y':10}
    #p1 = [610,440]
    '''Uncrop version '''
    p1 = [585,460]
    p4 = [373,623]
    p3 = [1000,623]
    p2 = [710,460]
    #p2 = [680,444]
    '''CROPPED'''
    # p1 = [310,35]
    # p4 = [75,205]
    # p3 = [750,205]
    # p2 = [490,35]
    #p2 = [680,444]


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
    return out,M