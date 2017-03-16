import pickle
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob


REPORT = True
CALIBRATE_CAMERA = False



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




def crop_image(img):
    x_crop = [250,1180]
    y_crop = [440,665]
    out = img[y_crop[0]:y_crop[1],x_crop[0]:x_crop[1]]
    return out



def flat_perspective(img):
    #points taken manualy in paint
    img_size = (600, 800)
    offset = {'x':80,'y':10}
    #p1 = [610,440]
    '''Uncrop version '''
    # p1 = [540,490]
    # p4 = [373,623]
    # p3 = [1000,623]
    # p2 = [775,490]
    #p2 = [680,444]
    '''CROPPED'''
    p1 = [310,35]
    p4 = [75,205]
    p3 = [750,205]
    p2 = [490,35]
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
    return out


S_threshold = (80, 255)
H_threshold = (15, 100)

def s_treshording(img):

    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    # H = hls[:, :, 0]
    #  L = hls[:, :, 1]
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

def hls_magnitude(img):

    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)

    H = hls[:, :, 0]
    L = hls[:, :, 1]
    S = hls[:, :, 2]


    L = L*1

    S = S*3


    hls[:, :, 1] = L
    hls[:, :, 2] = S
    H = H * 0.5
    hls[:, :, 0] = H

    return hls



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
    #gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    gray = img
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
#report flatening images

if REPORT:
    img = cv2.imread('report/sample3.png')
    crop = crop_image(img)
    cv2.imwrite("report/crop_sample3.png", crop)



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
    plt.imshow(out)
    plt.show()


    # images = glob.glob('report/sample*.png')
    # for image_path in images:
    #     img = cv2.imread(image_path)
    #     out = s_treshording(img)
    #     new_path = image_path.replace('sample','s_treshold_sample_')
    #     cv2.imwrite(new_path, out)



def remove_blobs(img,min_area,max_area):
    pass



#######FIND LINES FUNCTIONMS



def window_mask(width, height, img_ref, center, level):
    output = np.zeros_like(img_ref)
    output[int(img_ref.shape[0] - (level + 1) * height):int(img_ref.shape[0] - level * height),
    max(0, int(center - width / 2)):min(int(center + width / 2), img_ref.shape[1])] = 1
    return output


def find_window_centroids(image, window_width, window_height, margin):
    warped = image

    window_centroids = []  # Store the (left,right) window centroid positions per level
    window = np.ones(window_width)  # Create our window template that we will use for convolutions

    # First find the two starting positions for the left and right lane by using np.sum to get the vertical image slice
    # and then np.convolve the vertical image slice with the window template

    # Sum quarter bottom of image to get slice, could use a different ratio
    l_sum = np.sum(warped[int(3 * warped.shape[0] / 4):, :int(warped.shape[1] / 2)], axis=0)
    l_center = np.argmax(np.convolve(window, l_sum)) - window_width / 2
    r_sum = np.sum(warped[int(3 * warped.shape[0] / 4):, int(warped.shape[1] / 2):], axis=0)
    r_center = np.argmax(np.convolve(window, r_sum)) - window_width / 2 + int(warped.shape[1] / 2)

    # Add what we found for the first layer
    window_centroids.append((l_center, r_center))

    # Go through each layer looking for max pixel locations
    for level in range(1, (int)(warped.shape[0] / window_height)):
        # convolve the window into the vertical slice of the image
        image_layer = np.sum(
            warped[int(warped.shape[0] - (level + 1) * window_height):int(warped.shape[0] - level * window_height), :],
            axis=0)
        conv_signal = np.convolve(window, image_layer)
        # Find the best left centroid by using past left center as a reference
        # Use window_width/2 as offset because convolution signal reference is at right side of window, not center of window
        offset = window_width / 2
        l_min_index = int(max(l_center + offset - margin, 0))
        l_max_index = int(min(l_center + offset + margin, warped.shape[1]))
        l_center = np.argmax(conv_signal[l_min_index:l_max_index]) + l_min_index - offset
        # Find the best right centroid by using past right center as a reference
        r_min_index = int(max(r_center + offset - margin, 0))

        r_max_index = int(min(r_center + offset + margin, warped.shape[1]))
        r_center = np.argmax(conv_signal[r_min_index:r_max_index]) + r_min_index - offset
        # Add what we found for that layer

        window_centroids.append((l_center, r_center))
    return window_centroids




def find_line_peaks(img):
    warped = img

    window_width = 30
    window_height = 50  # Break image into 9 vertical layers since image height is 720
    margin = 10  # How much to slide left and right for searching

    window_centroids = find_window_centroids(warped, window_width, window_height, margin)

    measure_curvate(img, window_centroids)

    # If we found any window centers
    if len(window_centroids) > 0:

        # Points used to draw all the left and right windowsq
        l_points = np.zeros_like(warped)
        r_points = np.zeros_like(warped)

        offset_x = int(window_width/2)
        offset_y = int(window_height/2)

        # Go through each level and draw the windows
        for level in range(0, len(window_centroids)):
            # Window_mask is a function to draw window areas
            h_pos = int(window_height*level - offset_y)


            box_l = img[h_pos-offset_y:h_pos+offset_y,
                    window_centroids[level][0]-offset_x:window_centroids[level][0]+offset_x]
            box_r = img[h_pos - offset_y:h_pos + offset_y,
                    window_centroids[level][1] - offset_x:window_centroids[level][1] + offset_x]

            #skip points if no info
            if ( box_l.mean() > 0 ):
                l_mask = window_mask(window_width, window_height, warped, window_centroids[level][0], level)
                l_points[(l_points == 255) | ((l_mask == 1))] = 255
            if (box_r.mean() > 0):
                r_mask = window_mask(window_width, window_height, warped, window_centroids[level][1], level)
                r_points[(r_points == 255) | ((r_mask == 1))] = 255



        # Draw the results
        #template = np.array(r_points + l_points, np.uint8)  # add both left and right window pixels together

        zero_channel = np.zeros_like(r_points)  # create a zero color channle
        right_line = np.array(cv2.merge((zero_channel, r_points, zero_channel)), np.uint8)  # make window pixels green
        left_line = np.array(cv2.merge((l_points, zero_channel, zero_channel)), np.uint8)  # make window pixels green
        warpage = np.array(cv2.merge((warped, warped, warped)),
                           np.uint8)  # making the original road pixels 3 color channels
        output = cv2.addWeighted(right_line, 1, left_line, 1, 0)  # overlay the orignal road image with window results

    # If no window centers found, just display orginal road image
    else:
        output = np.array(cv2.merge((warped, warped, warped)), np.uint8)

    # Display the final results
    # plt.imshow(output)
    # plt.title('window fitting results')
    # plt.show()

    return output





def image_preprocesor(img):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    out = img
    #1 Fix camera disortion
    #out = cv2.undistort(img, camear_calibration['mtx'], camear_calibration['dist'], None, camear_calibration['mtx'])
    #out = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    #out = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #

    out2 = s_treshording(img)

    #out = mag_thresh(out, sobel_kernel=3, mag_thresh=(30, 100))
    #out = dir_threshold(out, sobel_kernel=5, thresh=(0.5, 1.2))
    #out = hls_magnitude(out)
    out1 = mag_thresh(out, sobel_kernel=5, mag_thresh=(50, 100))
    out = cv2.add(out1, out2)
    #out = dir_threshold(out, sobel_kernel=3, thresh=(0.7, 1.1))
    kernel = np.ones((3, 3), np.uint8)
    #
    out = cv2.dilate(out, kernel, iterations=1)
    # kernel = np.ones((3, 3), np.uint8)
    # out = cv2.erode(out, kernel, iterations=3)
    #
    #out = cv2.erode(out, kernel, iterations=1)
    #out = cv2.cvtColor(out, cv2.COLOR_HLS2RGB)
    #out = cv2.cvtColor(out, cv2.COLOR_RGB2GRAY)
    #ret, out = cv2.threshold(out, 0, 255, cv2.THRESH_OTSU)
    #out = cv2.adaptiveThreshold(out, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 25, -15)
    #out = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    out3 = clahe.apply(out)
    out = out & out3
    out = dir_threshold(out, sobel_kernel=5, thresh=(0.5, 1.2))

    return out


def measure_curvate(img, window_centroids):
    pass



# if True:
#     exit()

cap = cv2.VideoCapture('project_video.mp4')
#cap = cv2.VideoCapture('harder_challenge_video.mp4')
#cap = cv2.VideoCapture('challenge_video.mp4')

while(cap.isOpened()):

    ret, out = cap.read()
    in_img = out
    if ret==True:

        crop = crop_image(out)
        out = crop

        # write the flipped frame
        #out.write(frame)

        #out = s_treshording(frame)
        # out = dir_threshold(img)

        out = image_preprocesor(out)

        cv2.imshow('frame2', crop)
        out = flat_perspective(out)
        flat = out
        cv2.imshow('frame', flat)
        out = find_line_peaks(out)
        #out = cv2.addWeighted(out, 0.5, flat, 0.7,0)
        cv2.imshow('lines', out )


        cv2.imshow('xxx',in_img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.imwrite("report/sample_pre.png", in_img)
            cv2.imwrite("report/sample_out.png", out)
            break
    else:
        break

cap.release()
#cv2.destroyAllWindows()







