import cv2
import numpy as np


def s_binnary(img):
    S_channel = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)[:,:,2]
    threshold= [170,255]

    binary = np.zeros_like(S_channel)
    binary[(S_channel >= threshold[0]) & (S_channel <= threshold[1])] = 255

    return binary


# https://hidefcolor.com/blog/color-management/what-is-lab-color-space/
def b_binnary(img):
    B_channel = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)[:,:,0]
    threshold= [155,255]

    binary = np.zeros_like(B_channel)
    binary[(B_channel >= threshold[0]) & (B_channel <= threshold[1])] = 255

    return binary

clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

def white_binnary(img):

    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #img = clahe.apply(img)
    threshold= [245,255]
    #
    binary = np.zeros_like(img)
    binary[(img >= threshold[0]) & (img <= threshold[1])] = 1



    return binary

def combine_preprocesors(img):
    out1 = b_binnary(img)
    out2 = s_binnary(img)
    out3 = white_binnary(img)

    return ( out1 & out2 ) | out3

