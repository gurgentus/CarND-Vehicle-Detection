import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
#import helper as hlp
import os
import matplotlib.ticker as plticker
import tensorflow as tf
import glob
import time
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from skimage.feature import hog
# NOTE: the next import is only valid
# for scikit-learn version <= 0.17
# if you are using scikit-learn >= 0.18 then use this:
# from sklearn.model_selection import train_test_split
from sklearn.cross_validation import train_test_split
import pickle

flags = tf.app.flags
FLAGS = flags.FLAGS

# command line flags
# turns on various calibration and visualization options
flags.DEFINE_string('calibrate', False, "calibration flag")

# conversion factors from pixels to meters
ym_per_pix = 30/720 # meters per pixel in y dimension
xm_per_pix = 3.7/1100 # meters per pixel in x dimension
# these scales will be used in the curvature formula formula
# x = a f(by) to give radius of curvature in meters from its value in pixels
# (so don't have to recalculate in meters)
a = xm_per_pix
b = 1/ym_per_pix

# define global variables for transformations and left and right lane data
global M, src, dst, left_lane, right_lane

src = []
dst = []

# Define a class to receive the characteristics of each line detection
class Line():
    def __init__(self):
        # was the line detected in the last iteration?
        self.detected = False
        # number of times in a row detection failed
        self.num_errors = 0
        # x values of the last n fits of the line
        self.recent_xfitted = []
        # average x values of the fitted line over the last n iterations
        self.bestx = None
        self.recent_coefffit = []
        # polynomial coefficients averaged over the last n iterations
        self.best_fit = None
        # polynomial coefficients for the most recent fit
        self.current_fit = [np.array([False])]
        # radius of curvature of the line in some units
        self.radius_of_curvature = None
        # distance in meters of vehicle center from the line
        self.line_base_pos = None
        # difference in fit coefficients between last and new fits
        self.diffs = np.array([0,0,0], dtype='float')
        #x values for detected line pixels
        self.allx = None
        #y values for detected line pixels
        self.ally = None



#lesson functions
def convert_color(img, conv='RGB2YCrCb'):
    if conv == 'RGB2YCrCb':
        return cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    if conv == 'BGR2YCrCb':
        return cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    if conv == 'RGB2LUV':
        return cv2.cvtColor(img, cv2.COLOR_RGB2LUV)

def get_hog_features(img, orient, pix_per_cell, cell_per_block,
                        vis=False, feature_vec=True):
    # Call with two outputs if vis==True
    if vis == True:
        features, hog_image = hog(img, orientations=orient,
                                  pixels_per_cell=(pix_per_cell, pix_per_cell),
                                  cells_per_block=(cell_per_block, cell_per_block),
                                  transform_sqrt=False,
                                  visualise=vis, feature_vector=feature_vec)
        return features, hog_image
    # Otherwise call with one output
    else:
        features = hog(img, orientations=orient,
                       pixels_per_cell=(pix_per_cell, pix_per_cell),
                       cells_per_block=(cell_per_block, cell_per_block),
                       transform_sqrt=False,
                       visualise=vis, feature_vector=feature_vec)
        return features

def bin_spatial(img, size=(32, 32)):
    color1 = cv2.resize(img[:,:,0], size).ravel()
    color2 = cv2.resize(img[:,:,1], size).ravel()
    color3 = cv2.resize(img[:,:,2], size).ravel()
    return np.hstack((color1, color2, color3))

def color_hist(img, nbins=32):    #bins_range=(0, 256)
    # Compute the histogram of the color channels separately
    channel1_hist = np.histogram(img[:,:,0], bins=nbins)
    channel2_hist = np.histogram(img[:,:,1], bins=nbins)
    channel3_hist = np.histogram(img[:,:,2], bins=nbins)
    # Concatenate the histograms into a single feature vector
    hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))
    # Return the individual histograms, bin_centers and feature vector
    return hist_features


# Define a function to extract features from a list of images
# Have this function call bin_spatial() and color_hist()
def extract_features(imgs, color_space='RGB', spatial_size=(32, 32),
                        hist_bins=32, orient=9,
                        pix_per_cell=8, cell_per_block=2, hog_channel=0,
                        spatial_feat=True, hist_feat=True, hog_feat=True):
    # Create a list to append feature vectors to
    features = []
    # Iterate through the list of images
    for file in imgs:
        file_features = []
        # Read in each one by one
        image = mpimg.imread(file)
        # apply color conversion if other than 'RGB'
        if color_space != 'RGB':
            if color_space == 'HSV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            elif color_space == 'LUV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2LUV)
            elif color_space == 'HLS':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
            elif color_space == 'YUV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
            elif color_space == 'YCrCb':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
        else: feature_image = np.copy(image)

        if spatial_feat == True:
            spatial_features = bin_spatial(feature_image, size=spatial_size)
            file_features.append(spatial_features)
        if hist_feat == True:
            # Apply color_hist()
            hist_features = color_hist(feature_image, nbins=hist_bins)
            file_features.append(hist_features)
        if hog_feat == True:
        # Call get_hog_features() with vis=False, feature_vec=True
            if hog_channel == 'ALL':
                hog_features = []
                for channel in range(feature_image.shape[2]):
                    hog_features.append(get_hog_features(feature_image[:,:,channel],
                                        orient, pix_per_cell, cell_per_block,
                                        vis=False, feature_vec=True))
                hog_features = np.ravel(hog_features)
            else:
                hog_features = get_hog_features(feature_image[:,:,hog_channel], orient,
                            pix_per_cell, cell_per_block, vis=False, feature_vec=True)
            # Append the new feature vector to the features list
            file_features.append(hog_features)
        features.append(np.concatenate(file_features))
    # Return list of feature vectors
    return features


# this function calibrates the camera using the chessboard
# and returns camera calibration parameters
def calibrate_camera():
    chessboard_points = []
    image_points = []

    my_grid = np.zeros((5*9,3), np.float32)
    my_grid[:,:2] = np.mgrid[0:9, 0:5].T.reshape(-1,2)


    for fn in os.listdir('camera_cal/'):
        if fn[0]!='.':
            # read calibration image
            img = mpimg.imread('camera_cal/'+fn)

            # convert to grayscale
            gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            # obtain chessboard corners using cv2 helper function
            ret, corners = cv2.findChessboardCorners(gray, (9,5), None)
            # if successfull append to the chessboard array
            if ret == True:
                chessboard_points.append(my_grid)
                image_points.append(corners)
                # draw the corners for verification purposes
                #img = cv2.drawChessboardCorners(img, (9,5), corners, ret)
                #plt.imshow(img)
                #plt.show()

    # obtain matrix calibration properties based on two sets of points
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(chessboard_points, \
        image_points, gray.shape[::-1], None, None)

    # test the code by applying undistort to one of the calibration images
    image2 = mpimg.imread('camera_cal/calibration3.jpg')
    dst = cv2.undistort(image2, mtx, dist, None, mtx)
    cv2.imwrite('output_images/undistort_output.jpg', dst)

    # test the code by applying undistort to one of the car images
    image3 = mpimg.imread('test_images/test3.jpg')
    dst = cv2.undistort(image3, mtx, dist, None, mtx)
    dst = cv2.cvtColor(dst,cv2.COLOR_BGR2RGB)
    cv2.imwrite('output_images/test3_output.jpg', dst)

    return ret, mtx, dist

# Define a function that takes an image, number of x and y points,
# camera matrix and distortion coefficients
# and returns a top down view
# if calibrate flag is on allow the user to select points that determine
# the projection
def change_view(image, mtx, dist, calibrate=False):

    global M, src, dst, left_lane, right_lane

    # default mapping of four points from camera view to top down view
    if len(src) == 0:
        # define default destination points
        img_size = (image.shape[1], image.shape[0])
        woffset = 100
        src = [[568, 477], [758, 477], [1095, 679], [292, 679]]
        dst = [[woffset, woffset], [img_size[0]-woffset, woffset],
                                         [img_size[0]-woffset, img_size[1]-woffset],
                                         [woffset, img_size[1]-woffset]]

    src = np.array(src)
    dst = np.array(dst)

    # Given src and dst points, calculate the perspective transform matrix
    M = cv2.getPerspectiveTransform(np.float32(src), np.float32(dst))

    if calibrate:
        # if calibrate flag is on, will allow user to select source poitns in the 3d image
        fig = plt.figure(figsize=(20,10))
        ax = fig.add_subplot(111)
        ax.set_title('Select 4 points forming an undistored rectangle clockwise starting from top left')

        ax.set_yticks(np.arange(0,1000,30))

        ax.patch.set_alpha(0)
        implot = ax.imshow(image)
        # Add the grid
        ax.grid(which='major', axis='both',  color='0.65', linestyle='-')

        src = []
        # add a point to the src array when the user clicks on that point
        def onclick(event):
            if event.xdata != None and event.ydata != None:
                print(event.xdata, event.ydata)
                src.append([int(event.xdata), int(event.ydata)])
                print(src)
        cid = fig.canvas.mpl_connect('button_press_event', onclick)
        plt.show()
        src = np.array(src)
        # after the user closes the window src should have 4 pairs of points
        # Given src and dst points, calculate the perspective transform matrix
        M = cv2.getPerspectiveTransform(np.float32(src), np.float32(dst))

    # Use the OpenCV undistort() function to remove distortion
    undist = cv2.undistort(image, mtx, dist, None, mtx)
    img_size = (undist.shape[1], undist.shape[0])

    # Warp the image using OpenCV warpPerspective() to obtain top down view
    top_view = cv2.warpPerspective(undist, M, img_size, flags=cv2.INTER_LINEAR)

    if calibrate:
        # display and save the results if calibrate is on
        top_view_todraw = top_view.copy()
        cv2.polylines(image, [src], True, (0,255,255), 3)
        cv2.polylines(top_view_todraw, [dst], True, (0,255,255), 3)
        # Plot the result
        f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
        f.tight_layout()
        ax1.imshow(image)
        ax1.set_title('Original Image', fontsize=50)
        ax2.imshow(top_view_todraw)
        ax2.set_title('Top View', fontsize=50)
        plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
        plt.show()
        # save image
        road_top_view = cv2.cvtColor(top_view_todraw,cv2.COLOR_BGR2RGB)
        cv2.imwrite('output_images/road_top_view.jpg', road_top_view)

    return top_view, M

# this function detects the lanes and fills in the corresponding information
# in left_lane and right_lane Line objects that are passed into it
def detect_lanes(image, left_lane, right_lane, calibrate=False):

    if left_lane.detected and right_lane.detected:
        # if both lanes were detected in the previous frame
        # we search around those lanes for the current frame
        nonzero = image.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        margin = 50
        # obtain 'hot' pixels around the previously fitted lanes
        left_lane_inds = ((nonzerox > (left_lane.best_fit[0]*(nonzeroy**2) + \
            left_lane.best_fit[1]*nonzeroy + left_lane.best_fit[2] - margin)) \
            & (nonzerox < (left_lane.best_fit[0]*(nonzeroy**2) + \
            left_lane.best_fit[1]*nonzeroy + left_lane.best_fit[2] + margin)))
        right_lane_inds = ((nonzerox > (right_lane.best_fit[0]*(nonzeroy**2) + \
            right_lane.best_fit[1]*nonzeroy + right_lane.best_fit[2] - margin)) \
            & (nonzerox < (right_lane.best_fit[0]*(nonzeroy**2) + \
            right_lane.best_fit[1]*nonzeroy + right_lane.best_fit[2] + margin)))

    else:
        # if lanes were not detected last time
        # will follow the histogram approach from lectures
        # Take a histogram of the bottom half of the image
        histogram = np.sum(image[image.shape[0]/2:,:], axis=0)


        # Find the peak of the left and right halves of the histogram
        # These will be the starting point for the left and right lines
        midpoint = np.int(histogram.shape[0]/2)
        leftx_base = np.argmax(histogram[:midpoint])
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint

        # Choose the number of sliding windows
        nwindows = 9
        # Set height of windows
        window_height = np.int(image.shape[0]/nwindows)
        # Identify the x and y positions of all nonzero pixels in the image
        nonzero = image.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Current positions to be updated for each window
        leftx_current = leftx_base
        rightx_current = rightx_base
        # Set the width of the windows +/- margin
        margin = 300
        # Set minimum number of pixels found to recenter window
        minpix = 50
        # Create empty lists to receive left and right lane pixel indices
        left_lane_inds = []
        right_lane_inds = []

        # Step through the windows one by one
        for window in range(nwindows):
            # Identify window boundaries in x and y (and right and left)
            win_y_low = image.shape[0] - (window+1)*window_height
            win_y_high = image.shape[0] - window*window_height
            win_xleft_low = leftx_current - margin
            win_xleft_high = leftx_current + margin
            win_xright_low = rightx_current - margin
            win_xright_high = rightx_current + margin

            if calibrate:
                # Create an output image to draw on and  visualize the result
                out_img = np.dstack((image, image, image))*255
                # Draw the windows on the visualization image
                cv2.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),(20,255,20), 2)
                cv2.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),(20,255,20), 2)

            # Identify the nonzero pixels in x and y within the window
            good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
            good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
            # Append these indices to the lists
            left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)
            # If you found > minpix pixels, recenter next window on their mean position
            if len(good_left_inds) > minpix:
                leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
            if len(good_right_inds) > minpix:
                rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

        # Concatenate the arrays of indices
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)

    left_fitx = nonzerox[left_lane_inds]
    left_fity = nonzeroy[left_lane_inds]
    right_fitx = nonzerox[right_lane_inds]
    right_fity = nonzeroy[right_lane_inds]

    # set how many of the last frames to average for smoother results
    n = 10
    # Generate x and y values for plotting based on the fitted polynomial
    ploty = np.linspace(0, image.shape[0]-1, image.shape[0] )
    left_lane.ally = ploty
    right_lane.ally = ploty

    # check if any 'hot' pixels were obtained, in which case set detected flag
    # of the Line object to True and fir a quadratic polynomial
    if len(left_lane_inds)==0:
        left_lane.detected = False
    else:
        left_lane.detected = True
        prev_fit = left_lane.current_fit
        left_lane.current_fit = np.polyfit(left_fity, left_fitx, 2)
        left_lane.allx = left_lane.current_fit[0]*ploty**2 + \
            left_lane.current_fit[1]*ploty + left_lane.current_fit[2]
        left_lane.diff = prev_fit - left_lane.current_fit

    if len(right_lane_inds)==0:
        right_lane.detected = False
    else:
        right_lane.detected = True
        prev_fit = right_lane.current_fit
        right_lane.current_fit = np.polyfit(right_fity, right_fitx, 2)
        right_lane.allx = right_lane.current_fit[0]*ploty**2 + \
            right_lane.current_fit[1]*ploty + right_lane.current_fit[2]
        right_lane.diff = prev_fit - right_lane.current_fit

    # if both lanes were detected add the results to the array which keeps track
    # of the last n detections, also average out the last n detections to obtained
    # the displayed results
    if left_lane.detected and right_lane.detected:
        if left_lane.recent_coefffit == []:
            left_lane.recent_coefffit = left_lane.current_fit.reshape(1,-1)
            left_lane.best_fit = left_lane.current_fit
        else:
            left_lane.recent_coefffit = np.vstack((left_lane.recent_coefffit, left_lane.current_fit))
            left_lane.best_fit = np.average(left_lane.recent_coefffit, axis=0)

        if left_lane.recent_xfitted == []:
            left_lane.recent_xfitted = left_lane.allx.reshape(1,-1)
            left_lane.bestx = left_lane.allx
        else:
            left_lane.recent_xfitted = np.vstack((left_lane.recent_xfitted, left_lane.allx))
            left_lane.bestx = np.average(left_lane.recent_xfitted, axis=0)
        if right_lane.recent_coefffit == []:
            right_lane.recent_coefffit = right_lane.current_fit.reshape(1,-1)
            right_lane.best_fit = right_lane.current_fit
        else:
            right_lane.recent_coefffit = np.vstack((right_lane.recent_coefffit, right_lane.current_fit))
            right_lane.best_fit = np.average(right_lane.recent_coefffit, axis=0)

        if right_lane.recent_xfitted == []:
            right_lane.recent_xfitted = right_lane.allx.reshape(1,-1)
            right_lane.bestx = right_lane.allx
        else:
            right_lane.recent_xfitted = np.vstack((right_lane.recent_xfitted, right_lane.allx))
            right_lane.bestx = np.average(right_lane.recent_xfitted, axis=0)

    # remove results from the frame that is too old (before n frames)
    if len(left_lane.recent_xfitted) > n:
        left_lane.recent_xfitted = np.delete(left_lane.recent_xfitted, 0, axis=0)
    if len(left_lane.recent_coefffit) > n:
        left_lane.recent_coefffit = np.delete(left_lane.recent_coefffit, 0, axis=0)
    if len(right_lane.recent_xfitted) > n:
        right_lane.recent_xfitted = np.delete(right_lane.recent_xfitted, 0, axis = 0)
    if len(right_lane.recent_coefffit) > n:
        right_lane.recent_coefffit = np.delete(right_lane.recent_coefffit, 0, axis=0)


    # fit based on the smoothed results to obtain curvature and offset info
    yvalue = image.shape[0]
    xvalue_l = left_lane.best_fit[0]*yvalue**2 + left_lane.best_fit[1]*yvalue + left_lane.best_fit[2]
    xvalue_r = right_lane.best_fit[0]*yvalue**2 + right_lane.best_fit[1]*yvalue + right_lane.best_fit[2]

    # calculate the radius of curvature for x = a f(by), where a and b represent
    # conversion factors as above
    left_lane.radius_of_curvature = ((1+a*a*b*b*(2*left_lane.best_fit[0]*yvalue + \
    left_lane.best_fit[1])**2)**1.5)/np.absolute(2*left_lane.best_fit[0]*a*b*b)


    right_lane.radius_of_curvature = ((1+a*a*b*b*(2*right_lane.best_fit[0]*yvalue + \
    right_lane.best_fit[1])**2)**1.5)/np.absolute(2*right_lane.best_fit[0]*a*b*b)

    # calculate distances from left and right lanes
    left_lane.line_base_pos = xm_per_pix*(image.shape[1]/2 - xvalue_l)
    right_lane.line_base_pos = xm_per_pix*(image.shape[1]/2 - xvalue_r)

    # sanity check, if this fails more than 5 times will lane detected is set to False
    # so on next iteration the lanes are searched from nothing
    # and all of the recent results cleared, basically treating the next frame as beginning
    if (np.absolute(left_lane.radius_of_curvature - right_lane.radius_of_curvature) > 1000000 \
       or left_lane.line_base_pos - right_lane.line_base_pos < 2):

        left_lane.num_errors = left_lane.num_errors + 1
        right_lane.num_errors = right_lane.num_errors + 1
        # remove bad data
        np.delete(left_lane.recent_coefffit, len(left_lane.recent_coefffit)-1, axis=0)
        np.delete(left_lane.recent_xfitted, len(left_lane.recent_xfitted)-1, axis=0)
        np.delete(right_lane.recent_coefffit, len(right_lane.recent_coefffit)-1, axis=0)
        np.delete(right_lane.recent_xfitted, len(right_lane.recent_xfitted)-1, axis=0)
        left_lane.detected = True
        right_lane.detected = True
        if left_lane.num_errors > 5 or right_lane.num_errors > 5:
            left_lane.detected = False
            right_lane.detected = False
            left_lane.num_errors = 0
            right_lane.num_errors = 0

    # if calibrate is set to true visualize the results
    if calibrate:
        out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
        out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]
        plt.figure()
        plt.imshow(out_img)
        plt.plot(left_lane.bestx, ploty, color='yellow')
        plt.plot(right_lane.bestx, ploty, color='blue')
        plt.xlim(0, 1280)
        plt.ylim(720, 0)
        plt.savefig('output_images/polyfit2.png')
        plt.show()
        out_img = cv2.cvtColor(out_img,cv2.COLOR_BGR2RGB)
        cv2.imwrite('output_images/polyfit.jpg', out_img)


# Define a single function that can extract features using hog sub-sampling and make predictions
def find_cars(img, ystart, ystop, scale, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins):

    draw_img = np.copy(img)
    img = img.astype(np.float32)/255

    img_tosearch = img[ystart:ystop,:,:]
    ctrans_tosearch = convert_color(img_tosearch, conv='RGB2YCrCb')
    if scale != 1:
        imshape = ctrans_tosearch.shape
        ctrans_tosearch = cv2.resize(ctrans_tosearch, (np.int(imshape[1]/scale), np.int(imshape[0]/scale)))

    ch1 = ctrans_tosearch[:,:,0]
    ch2 = ctrans_tosearch[:,:,1]
    ch3 = ctrans_tosearch[:,:,2]

    # Define blocks and steps as above
    nxblocks = (ch1.shape[1] // pix_per_cell)-1
    nyblocks = (ch1.shape[0] // pix_per_cell)-1
    nfeat_per_block = orient*cell_per_block**2
    # 64 was the orginal sampling rate, with 8 cells and 8 pix per cell
    window = 64
    nblocks_per_window = (window // pix_per_cell)-1
    cells_per_step = 2  # Instead of overlap, define how many cells to step
    nxsteps = (nxblocks - nblocks_per_window) // cells_per_step
    nysteps = (nyblocks - nblocks_per_window) // cells_per_step

    # Compute individual channel HOG features for the entire image
    hog1 = get_hog_features(ch1, orient, pix_per_cell, cell_per_block, feature_vec=False)
    hog2 = get_hog_features(ch2, orient, pix_per_cell, cell_per_block, feature_vec=False)
    hog3 = get_hog_features(ch3, orient, pix_per_cell, cell_per_block, feature_vec=False)

    for xb in range(nxsteps):
        for yb in range(nysteps):
            ypos = yb*cells_per_step
            xpos = xb*cells_per_step
            # Extract HOG for this patch
            hog_feat1 = hog1[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel()
            hog_feat2 = hog2[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel()
            hog_feat3 = hog3[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel()
            hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))

            xleft = xpos*pix_per_cell
            ytop = ypos*pix_per_cell

            # Extract the image patch
            subimg = cv2.resize(ctrans_tosearch[ytop:ytop+window, xleft:xleft+window], (64,64))

            # Get color features
            spatial_features = bin_spatial(subimg, size=spatial_size)
            hist_features = color_hist(subimg, nbins=hist_bins)

            # Scale features and make a prediction
            test_features = X_scaler.transform(np.hstack((spatial_features, hist_features, hog_features)).reshape(1, -1))
            #test_features = X_scaler.transform(np.hstack((shape_feat, hist_feat)).reshape(1, -1))
            test_prediction = svc.predict(test_features)

            if test_prediction == 1:
                xbox_left = np.int(xleft*scale)
                ytop_draw = np.int(ytop*scale)
                win_draw = np.int(window*scale)
                cv2.rectangle(draw_img,(xbox_left, ytop_draw+ystart),(xbox_left+win_draw,ytop_draw+win_draw+ystart),(0,0,255),6)

    return draw_img


# this is the function called by clip.fl_image
# process each frame of the video by calling process_image
# which allows a calibrate flag
def process_video(image):
    #result = process_image(image, calibrate=False)
    result = find_cars(image, ystart, ystop, scale, svc, X_scaler, \
        orient, pix_per_cell, cell_per_block, spatial_size, hist_bins)

    return result

# this funciton processes the image and locates the left and right lanes
# if calibrate flag is True the results are visualized
# (typical use is to turn this on for a test image, before processing the video)
def process_image(image_before, calibrate=False):

    # Choose a Sobel kernel size
    ksize = 3 # Choose a larger odd number to smooth gradient measurements

    # first we find edges based on the grayscale version of the image
    gray = cv2.cvtColor(image_before, cv2.COLOR_RGB2GRAY)


    # the four functions below collect information about x-gradient,
    # y-gradient, gradient magnitude and orientation and apply thresholding values
    # specified by thres to filter out certain values
    gradx = abs_sobel_thresh(gray, orient='x', sobel_kernel=ksize, thresh=(40, 155))
    grady = abs_sobel_thresh(gray, orient='y', sobel_kernel=ksize, thresh=(40, 155))
    mag_binary = mag_thresh(gray, sobel_kernel=ksize, thresh=(80, 155))
    dir_binary = dir_threshold(gray, sobel_kernel=ksize, thresh=(0, np.pi/4))
    filtered_gray = np.zeros_like(dir_binary)
    filtered_gray[((gradx == 1) & (grady == 1)) | ((mag_binary == 1) & (dir_binary == 1))] = 1

    # next we also find edges based on the s channel in hsv version of the image
    hsv = cv2.cvtColor(image_before, cv2.COLOR_RGB2HLS).astype(np.float)

    #l_channel = hsv[:,:,1]
    s_channel = hsv[:,:,2]

    # the four functions below collect information about x-gradient,
    # y-gradient, gradient magnitude and orientation and apply thresholding values
    # specified by thres to filter out certain values
    gradx = abs_sobel_thresh(s_channel, orient='x', sobel_kernel=ksize, thresh=(40, 155))
    grady = abs_sobel_thresh(s_channel, orient='y', sobel_kernel=ksize, thresh=(40, 155))
    mag_binary = mag_thresh(s_channel, sobel_kernel=ksize, thresh=(80, 155))
    dir_binary = dir_threshold(s_channel, sobel_kernel=ksize, thresh=(0, np.pi/4))
    filtered_color = np.zeros_like(dir_binary)
    filtered_color[((gradx == 1) & (grady == 1)) | ((mag_binary == 1) & (dir_binary == 1))] = 1

    # combine the two filtered images
    color_binary = np.dstack((np.zeros_like(filtered_gray), filtered_gray, filtered_color))
    combined_binary = np.zeros_like(filtered_color)
    combined_binary[(filtered_gray == 1) | (filtered_color == 1)] = 1

    # combination of the binary images generated from grayscale and hsv thresholding
    all_channels =  np.uint8(np.dstack((combined_binary, combined_binary, combined_binary))*255)

    global mtx, dist

    # generate the top view of the lane based on calibration done in the first step
    top_view, M = change_view(all_channels, mtx, dist, calibrate)

    if calibrate:
        top_view_img, M = change_view(image_before, mtx, dist, calibrate=False)
        cv2.imwrite('output_images/top_view_img.jpg', cv2.cvtColor(top_view_img,cv2.COLOR_BGR2RGB))

    # fill in lane information
    # from lane detection in detect_lanes function

    detect_lanes(top_view[:,:,0], left_lane, right_lane, calibrate)

    # Create an image to draw the lines on
    top_view_canvas = np.zeros_like(top_view).astype(np.uint8)

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_lane.bestx, left_lane.ally]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_lane.bestx, right_lane.ally])))])
    pts = np.hstack((pts_left, pts_right))

    result = image_before
    if left_lane.detected and right_lane.detected:
        # Draw the lane onto the warped blank image
        cv2.fillPoly(top_view_canvas, np.int_([pts]), (0,255, 0))

        # only fill in the bottom half of the image, otherwise if lane curves too much
        # it will not transfer back correctly
        #top_view_canvas[:top_view_canvas.shape[0]/4,:]=0
        # Warp the blank back to original image space using inverse perspective matrix (Minv)
        newwarp = cv2.warpPerspective(top_view_canvas, np.linalg.inv(M), (image_before.shape[1], image_before.shape[0]))


        # Combine the result with the original image
        result = cv2.addWeighted(image_before, 1, newwarp, 0.3, 0)

        font = cv2.FONT_HERSHEY_SIMPLEX
        text = 'left radius of curvature: ' + str(left_lane.radius_of_curvature) + ' m'
        text2 = 'right radius of curvature: ' + str(right_lane.radius_of_curvature) + ' m'

        offset = (right_lane.line_base_pos+left_lane.line_base_pos)/2
        text3 = 'offset from center: ' + str(offset) + ' m'
        # text = 'average x:' + str(left_lane.bestx) +  'latest coeff:' \
        #     + str(left_lane.current_fit[0]) + 'avg coeff:' + str(left_lane.best_fit[0])
        cv2.putText(result, text ,(10,30), font, 1, (0,0,0),2,cv2.LINE_AA)
        cv2.putText(result, text2 ,(10,60), font, 1, (0,0,0),2,cv2.LINE_AA)
        cv2.putText(result, text3 ,(10,90), font, 1, (0,0,0),2,cv2.LINE_AA)
    else:
        result = hlp.process_img(image_before)

    if calibrate:
        fig = plt.figure(figsize=(20,10))
        plt.imshow(result)
        plt.show()
        cv2.imwrite('output_images/result.jpg', cv2.cvtColor(result,cv2.COLOR_BGR2RGB))

    return result

from moviepy.editor import VideoFileClip


# Read in cars and notcars
#images = glob.glob('*.jpeg')
cars = []
notcars = []
# for image in images:
#     if 'image' in image or 'extra' in image:
#         notcars.append(image)
#     else:
#         cars.append(image)

for fn in os.listdir('vehicles/KITTI_extracted/'):
    if fn[0]!='.':
        img = 'vehicles/KITTI_extracted/'+fn
        cars.append(img)

for fn in os.listdir('vehicles/GTI_far'):
    if fn[0]!='.':
        img = 'vehicles/GTI_far/'+fn
        cars.append(img)

for fn in os.listdir('vehicles/GTI_Left'):
    if fn[0]!='.':
        img = 'vehicles/GTI_Left/'+fn
        cars.append(img)

for fn in os.listdir('vehicles/GTI_MiddleClose'):
    if fn[0]!='.':
        img = 'vehicles/GTI_MiddleClose/'+fn
        cars.append(img)

for fn in os.listdir('vehicles/GTI_Right'):
    if fn[0]!='.':
        img = 'vehicles/GTI_Right/'+fn
        cars.append(img)

for fn in os.listdir('non-vehicles/Extras/'):
    if fn[0]!='.':
        img = 'non-vehicles/Extras/'+fn
        notcars.append(img)

for fn in os.listdir('non-vehicles/GTI/'):
    if fn[0]!='.':
        img = 'non-vehicles/GTI/'+fn
        notcars.append(img)

# Reduce the sample size because
# The quiz evaluator times out after 13s of CPU time
#sample_size = 30000
#cars = cars[0:sample_size]
#notcars = notcars[0:sample_size]
print("cars:", len(cars), "not cars:", len(notcars))

color_space = 'HSV' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
orient = 10  # HOG orientations
pix_per_cell = 8 # HOG pixels per cell
cell_per_block = 2 # HOG cells per block
hog_channel = "ALL" # Can be 0, 1, 2, or "ALL"
spatial_size = (16, 16) # Spatial binning dimensions
hist_bins = 16    # Number of histogram bins
spatial_feat = True # Spatial features on or off
hist_feat = True # Histogram features on or off
hog_feat = True # HOG features on or off
y_start_stop = [300, 656] # Min and max in y to search in slide_window()


ystart = 300
ystop = 656
scale = 1.5

full = True

if full:

    car_features = extract_features(cars, color_space=color_space,
                            spatial_size=spatial_size, hist_bins=hist_bins,
                            orient=orient, pix_per_cell=pix_per_cell,
                            cell_per_block=cell_per_block,
                            hog_channel=hog_channel, spatial_feat=spatial_feat,
                            hist_feat=hist_feat, hog_feat=hog_feat)
    notcar_features = extract_features(notcars, color_space=color_space,
                            spatial_size=spatial_size, hist_bins=hist_bins,
                            orient=orient, pix_per_cell=pix_per_cell,
                            cell_per_block=cell_per_block,
                            hog_channel=hog_channel, spatial_feat=spatial_feat,
                            hist_feat=hist_feat, hog_feat=hog_feat)

    X = np.vstack((car_features, notcar_features)).astype(np.float64)
    # Fit a per-column scaler
    X_scaler = StandardScaler().fit(X)
    # Apply the scaler to X
    scaled_X = X_scaler.transform(X)

    # Define the labels vector
    y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))

    # Split up data into randomized training and test sets
    rand_state = np.random.randint(0, 100)
    X_train, X_test, y_train, y_test = train_test_split(
        scaled_X, y, test_size=0.2, random_state=rand_state)

    print('Using:',orient,'orientations',pix_per_cell,
        'pixels per cell and', cell_per_block,'cells per block')
    print('Feature vector length:', len(X_train[0]))
    # Use a linear SVC
    svc = LinearSVC()
    # Check the training time for the SVC
    t=time.time()
    svc.fit(X_train, y_train)

    t2 = time.time()
    print(round(t2-t, 2), 'Seconds to train SVC...')
    # Check the score of the SVC
    print('Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 4))
    # Check the prediction time for a single sample
    t=time.time()

    with open('modelHLV16.pickle', 'wb') as handle:
        pickle.dump(svc,handle)
    with open('scalerHLV16.pickle', 'wb') as handle:
        pickle.dump(X_scaler,handle)
else:
    with open('modelHLS.pickle', 'rb') as handle:
        svc = pickle.load(handle)
    with open('scalerHLS.pickle', 'rb') as handle:
        X_scaler = pickle.load(handle)

image = mpimg.imread('test_images/test1.jpg')
out_img = process_video(image)

plt.imshow(out_img)
plt.show()

# car_ind = np.random.randint(0, len(cars))
# car_image =mping.imread(cars[car_ind])

# output_video = 'detect1.mp4'
# clip1 = VideoFileClip("project_video.mp4")
# clip1 = VideoFileClip('project_video.mp4').subclip(10,25)
# processed_clip = clip1.fl_image(process_video)
# processed_clip.write_videofile(output_video, audio=False)
#
#




# # first calibrate the camera
# ret, mtx, dist = calibrate_camera()
#
# # will keep track of detected lane information
# # left_lane = Line()
# # right_lane = Line()
#
#
# # we will now apply thresholding to a test_images
# # image, this also calibrates the top-down transform
# # warning: the region chosen for calibration should correspond to
# # straight lines
# image_before = mpimg.imread('test_images/test1.jpg')
# process_image(image_before, calibrate=FLAGS.calibrate)
#
# # will keep track of detected lane information
# # reset for the video
# # left_lane = Line()
# # right_lane = Line()
#
# output_video = 'detect1.mp4'
# clip1 = VideoFileClip("project_video.mp4")
# #clip1 = VideoFileClip('project_video.mp4').subclip(0,8)
# processed_clip = clip1.fl_image(process_video)
# processed_clip.write_videofile(output_video, audio=False)
