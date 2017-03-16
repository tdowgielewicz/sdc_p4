import cv2
import numpy as np
import matplotlib.pyplot as plt


def s_binnary(img):
    S_channel = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2HLS)[:,:,2]
    threshold= [150,255]

    binary = np.zeros_like(S_channel)
    binary[(S_channel >= threshold[0]) & (S_channel <= threshold[1])] = 255

    return binary

def dir_combined(img2):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    out = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    S_channel = cv2.cvtColor(img2.astype(np.uint8), cv2.COLOR_BGR2HLS)[:, :, 2]

    out3 = clahe.apply(out)

    out1 = mag_thresh(img2, sobel_kernel=5, mag_thresh=(50, 100))

    out = out1 & out3


    out = dir_threshold(out, sobel_kernel=5, thresh=(0.5, 1.2))

    kernel = np.ones((3, 3), np.uint8)
    #
    out = cv2.dilate(out, kernel, iterations=2)

    binary = np.zeros_like(S_channel)  #uggly walkaround to have same type of images
    #binary[(img >= threshold[0]) & (img <= threshold[1])] = 255
    binary[(out >= 100) ] =255
    #
    # plt.imshow(out)
    # plt.show()

    return binary

def dir_threshold(img, sobel_kernel=3, thresh=(0, np.pi/2)):
    # Grayscale
    #gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    gray = img
    # Calculate the x and y gradients
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # Take the absolute value of the gradient direction,
    # apply a threshold, and create a binary image result
    absgraddir = np.arctan2(np.absolute(sobely), np.absolute(sobelx))
    binary_output =  np.zeros_like(absgraddir)
    binary_output[(absgraddir >= thresh[0]) & (absgraddir <= thresh[1])] = 255

    # Return the binary image
    return binary_output

def mag_thresh(img, sobel_kernel=3, mag_thresh=(0, 255)):
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Take both Sobel x and y gradients
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # Calculate the gradient magnitude
    gradmag = np.sqrt(sobelx**2 + sobely**2)
    # Rescale to 8 bit
    scale_factor = np.max(gradmag)/255
    gradmag = (gradmag/scale_factor).astype(np.uint8)
    # Create a binary image of ones where threshold is met, zeros otherwise
    binary_output = np.zeros_like(gradmag)
    binary_output[(gradmag >= mag_thresh[0]) & (gradmag <= mag_thresh[1])] = 255

    # Return the binary image
    return binary_output


# https://hidefcolor.com/blog/color-management/what-is-lab-color-space/
def b_binnary(img):
    B_channel = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2LAB)[:,:,0]
    threshold= [155,255]

    binary = np.zeros_like(B_channel)
    binary[(B_channel >= threshold[0]) & (B_channel <= threshold[1])] = 255

    return binary

clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

def white_binnary(img):

    img = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2GRAY)
    threshold= [225,255]
    #
    binary = np.zeros_like(img)
    binary[(img >= threshold[0]) & (img <= threshold[1])] = 255
    return binary


def combine_preprocesors(img):
    out1 = b_binnary(img)
    out2 = s_binnary(img)
    out3 = white_binnary(img)

    #out4 = dir_combined(img)

    # plt.imshow(out4)
    # plt.show()


    #add =  out3 | out4[:,:,2]
    #return out3
    return ( out1 & out2 ) | out3

