import numpy
import cv2
import glob
import matplotlib.pyplot as plt
import time
import numpy as np

def s_treshording(img):

    hls = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # plt.imshow(hls)
    # plt.show()


    return hls
#import skvideo.io
#cap = skvideo.io.VideoCapture

#So hardcore aproach to run movie without opencv ffmpeg build
#  ffmpeg -i "challenge_video.mp4" "frames/out-%03d.jpg"


OUTPUT_DIR = 'out_video'

images = sorted(glob.glob('frames/out-*.jpg'))
for x in images:
    img = cv2.imread(x)
    out = s_treshording(img)


    cv2.imwrite(x.replace('frames',OUTPUT_DIR),out )


#than on the end compile images to video
# ffmpeg -r 24 -f image2  -i out-%03d.jpg -vcodec libx264 -crf 25  -pix_fmt yuv420p test.mp4

