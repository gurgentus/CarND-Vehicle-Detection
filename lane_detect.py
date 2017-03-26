import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import helper as hlp
import os
import matplotlib.ticker as plticker
import tensorflow as tf
#import pickle

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

# the next three functions produce a binary image of 'hot' pixels based on
# x and y gradient thresholding, gradient magnitude thresholding, and
# gradient direction thresholding
# these are based on examples from lecture
def abs_sobel_thresh(channel, orient='x', sobel_kernel=3, thresh=(0, 255)):
    # Calculate directional gradient
    if orient == 'x':
        abs_sobel = np.absolute(cv2.Sobel(channel, cv2.CV_64F, 1, 0, ksize=sobel_kernel))
    if orient == 'y':
        abs_sobel = np.absolute(cv2.Sobel(channel, cv2.CV_64F, 0, 1, ksize=sobel_kernel))
    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
    grad_binary = np.zeros_like(scaled_sobel)
    # Apply threshold
    grad_binary[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1
    return grad_binary

def mag_thresh(channel, sobel_kernel=3, thresh=(0, 255)):
    # Calculate gradient magnitude
    sobelx = cv2.Sobel(channel, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(channel, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # Calculate the gradient magnitude
    magnitude = np.sqrt(sobelx**2 + sobely**2)
    # Rescale to 8 bit
    scale_factor = np.max(magnitude)/255
    magnitude = (magnitude/scale_factor).astype(np.uint8)
    # Create a binary image of ones where threshold is met, zeros otherwise
    mag_binary = np.zeros_like(magnitude)
    # Apply threshold
    mag_binary[(magnitude >= thresh[0]) & (magnitude <= thresh[1])] = 1

    return mag_binary

def dir_threshold(channel, sobel_kernel=3, thresh=(0, np.pi/2)):
    # Calculate gradient direction
    sobelx = cv2.Sobel(channel, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(channel, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    direction = np.arctan2(np.absolute(sobely), np.absolute(sobelx))
    dir_binary =  np.zeros_like(direction)
    # Apply threshold
    dir_binary[(direction >= thresh[0]) & (direction <= thresh[1])] = 1

    return dir_binary

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
        break
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



# this is the function called by clip.fl_image
# process each frame of the video by calling process_image
# which allows a calibrate flag
def process_video(image):
    result = process_image(image, calibrate=False)
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

        left_lane.radius_of_curvature = "%.2f" % round(left_lane.radius_of_curvature,2)
        right_lane.radius_of_curvature = "%.2f" % round(right_lane.radius_of_curvature,2)

        font = cv2.FONT_HERSHEY_SIMPLEX
        text = 'left radius of curvature: ' + str(left_lane.radius_of_curvature) + ' m'
        text2 = 'right radius of curvature: ' + str(right_lane.radius_of_curvature) + ' m'

        offset = (right_lane.line_base_pos+left_lane.line_base_pos)/2
        offset = "%.2f" % round(offset,2)

        text3 = 'offset from center: ' + str(offset) + ' m'
        # text = 'average x:' + str(left_lane.bestx) +  'latest coeff:' \
        #     + str(left_lane.current_fit[0]) + 'avg coeff:' + str(left_lane.best_fit[0])
        cv2.putText(result, text ,(10,30), font, 1, (255,255,255),2,cv2.LINE_AA)
        cv2.putText(result, text2 ,(10,60), font, 1, (255,255,255),2,cv2.LINE_AA)
        cv2.putText(result, text3 ,(10,90), font, 1, (255,255,255),2,cv2.LINE_AA)
    else:
        result = hlp.process_img(image_before)

    if calibrate:
        fig = plt.figure(figsize=(20,10))
        plt.imshow(result)
        plt.show()
        cv2.imwrite('output_images/result.jpg', cv2.cvtColor(result,cv2.COLOR_BGR2RGB))

    return result

from moviepy.editor import VideoFileClip

# first calibrate the camera
ret, mtx, dist = calibrate_camera()


# will keep track of detected lane information
# reset for the video
left_lane = Line()
right_lane = Line()
