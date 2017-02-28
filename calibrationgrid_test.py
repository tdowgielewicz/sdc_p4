import pickle
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob

# Read in the saved objpoints and imgpoints

# Read an calibration image
nx = 9
ny = 6

img = cv2.imread('camera_calibration/calibration3.jpg')
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

#plt.show()

#chesboard paramiters

objpoints = []
imgpoints = []

objp = np.zeros((nx*ny,3),np.float32)

print(objp)
objp[:,:2] = np.mgrid[0:nx,0:ny].T.reshape(-1,2)

print("AHHH",objp)
ret, corners = cv2.findChessboardCorners(gray, (nx,ny), None)

if ret:
    imgpoints.append(corners)
    objpoints.append(objp)


img = cv2.drawChessboardCorners(img, (nx,ny), corners, ret)

ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

dst = cv2.undistort(img, mtx, dist, None, mtx)

plt.imshow(img,cmap='Greys_r')
plt.show()


# # TODO: Write a function that takes an image, object points, and image points
# # performs the camera calibration, image distortion correction and
# # returns the undistorted image
# def cal_undistort(img, objpoints, imgpoints):
#     # Use cv2.calibrateCamera() and cv2.undistort()
#     undist = np.copy(img)  # Delete this line
#     return undist
#
# undistorted = cal_undistort(img, objpoints, imgpoints)
#
# f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
# f.tight_layout()
# ax1.imshow(img)
# ax1.set_title('Original Image', fontsize=50)
# ax2.imshow(undistorted)
# ax2.set_title('Undistorted Image', fontsize=50)
# plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)