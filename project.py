import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from moviepy.editor import VideoFileClip
import numpy as np
import cv2
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
from scipy.ndimage.measurements import label

import pickle
import lane_detect as ld


flags = tf.app.flags
FLAGS = flags.FLAGS

# command line flags
# turns on various calibration and visualization options
flags.DEFINE_string('visualize', False, "visualization flag")
visualize = FLAGS.visualize

global heat_recent # will be used to store n recent heatmaps

# lesson functions
# converts color space
def convert_color(img, conv='RGB2YCrCb'):
    if conv == 'RGB2YCrCb':
        return cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    if conv == 'BGR2YCrCb':
        return cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    if conv == 'RGB2LUV':
        return cv2.cvtColor(img, cv2.COLOR_RGB2LUV)

# extract HOG features using skimage hog functions
# pass number of orientation bins for binning Gradients
# size of the cell by specifying how many pixels per cell
# size of the block by specifying number of cells per block
# vis=True will produce visualizaiton pictures
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

# resize the image to given size and create a single horizontal array
# feature vector consisting of values from the three channels
def bin_spatial(img, size=(32, 32)):
    color1 = cv2.resize(img[:,:,0], size).ravel()
    color2 = cv2.resize(img[:,:,1], size).ravel()
    color3 = cv2.resize(img[:,:,2], size).ravel()
    return np.hstack((color1, color2, color3))

# create a single feature vector consisting of histogram distribution
# of intentisities from the three channels
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
# This function takes an image and extracts the arrays of the three
# types of features by calling the above three functions
# those arrays are concatinated into a single feature vector
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
                                        vis=visualize, feature_vec=True))
                hog_features = np.ravel(hog_features)
            else:
                hog_features = get_hog_features(feature_image[:,:,hog_channel], orient,
                            pix_per_cell, cell_per_block, vis=visualize, feature_vec=True)
            # Append the new feature vector to the features list
            file_features.append(hog_features)
        features.append(np.concatenate(file_features))
    # Return list of feature vectors
    return features


# From lesson.  Defines a single function that can extract features using hog sub-sampling and make predictions
def find_cars(img, bbox_list, ystart, ystop, scale, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins):

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
                # append positive prediction to the list of bounding boxes
                bbox_list.append([(xbox_left, ytop_draw+ystart),(xbox_left+win_draw,ytop_draw+win_draw+ystart)])

# update the heatmap by adding 1 to any pixel overlapping with a detected bounding box
def add_heat(heatmap, bbox_list):
    # Iterate through list of bboxes
    for box in bbox_list:
        # Add += 1 for all pixels inside each bbox
        # Assuming each "box" takes the form ((x1, y1), (x2, y2))
        heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1

    # Return updated heatmap
    return heatmap# Iterate through list of bboxes

def apply_threshold(heatmap, threshold):
    # Zero out pixels below the threshold
    heatmap[heatmap <= threshold] = 0
    # Return thresholded map
    return heatmap

def draw_labeled_bboxes(img, labels):
    # Iterate through all detected cars
    for car_number in range(1, labels[1]+1):
        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Define a bounding box based on min/max x and y
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
        # Draw the box on the image
        cv2.rectangle(img, bbox[0], bbox[1], (0,0,255), 6)
    # Return the image
    return img

# this is the function called by clip.fl_image
# process each frame of the video by calling process_image
# which allows a calibrate flag
def process_video(image):

    global heat_recent
    heat = np.zeros_like(image[:,:,0]).astype(np.float)

    box_list = []

    result = find_cars(image, box_list, 400, 500, 1.2, svc, X_scaler, \
        orient, pix_per_cell, cell_per_block, spatial_size, hist_bins)
    result = find_cars(image, box_list, 400, 550, 1.8, svc, X_scaler, \
        orient, pix_per_cell, cell_per_block, spatial_size, hist_bins)
    # result = find_cars(image, box_list, 400, 800, 2.3, svc, X_scaler, \
    #     orient, pix_per_cell, cell_per_block, spatial_size, hist_bins)

    # Add heat to each box in box list
    heat = add_heat(heat,box_list)

    if heat_recent == []:
        heat_recent = heat.reshape(1, heat.shape[0], heat.shape[1])
    else:
        heat = heat.reshape(1, heat.shape[0], heat.shape[1])
        heat_recent = np.concatenate((heat_recent, heat), axis=0)
        if len(heat_recent) > 8:
            heat_recent = np.delete(heat_recent, 0, axis=0)
        heat = np.sum(heat_recent, axis=0)

    # Apply threshold to help remove false positives
    # if visualize is on, do this based on only one frame, so lower thr
    if visualize:
        thr = 1
    else:
        thr = 12
    heat = apply_threshold(heat,thr)

    # Visualize the heatmap when displaying
    heatmap = np.clip(heat, 0, 255)

    # Find final boxes from heatmap using label function
    labels = label(heatmap)

    image_lanes = ld.process_image(image, calibrate=False)
    result = draw_labeled_bboxes(np.copy(image_lanes), labels)

    if visualize:
        fig = plt.figure()
        plt.subplot(121)
        plt.imshow(result)
        plt.title('Car Positions')
        plt.subplot(122)

        plt.imshow(heatmap, cmap='hot')
        plt.title('Heat Map')
        fig.tight_layout()
        plt.savefig('output_images/heatmap.png')
        plt.show()

    return result


cars = []
notcars = []
# load images for training/validation
for fn in os.listdir('vehicles/KITTI_extracted/'):
    if fn[0]!='.':
        img = 'vehicles/KITTI_extracted/'+fn
        cars.append(img)

for fn in os.listdir('vehicles/GTI_Far'):
    if fn[0]!='.':
        img = 'vehicles/GTI_Far/'+fn
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

print("cars:", len(cars), "not cars:", len(notcars))

# set various parameters
color_space = 'YCrCb' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
orient = 10  # HOG orientations
pix_per_cell = 8 # HOG pixels per cell
cell_per_block = 2 # HOG cells per block
hog_channel = "ALL" # Can be 0, 1, 2, or "ALL"
spatial_size = (64, 64) # Spatial binning dimensions
hist_bins = 16    # Number of histogram bins
spatial_feat = True # Spatial features on or off
hist_feat = True # Histogram features on or off
hog_feat = True # HOG features on or off

# visualize random image in data set
if visualize:
    car_ind = np.random.randint(0, len(cars))
    car_image = mpimg.imread(cars[car_ind])
    ncar_ind = np.random.randint(0, len(notcars))
    ncar_image = mpimg.imread(notcars[ncar_ind])
    fig = plt.figure()
    plt.subplot(121)
    plt.imshow(car_image)
    plt.title('Random car image')
    plt.subplot(122)
    plt.imshow(ncar_image)
    plt.title('Random not car image')
    fig.tight_layout()
    plt.savefig('output_images/random_image.png')
    plt.show()
    # visualize hog feature extraction
    hog1, hog_image = get_hog_features(car_image[:,:,0],
                        orient, pix_per_cell, cell_per_block,
                        vis=True, feature_vec=True)
    fig = plt.figure()
    plt.subplot(121)
    plt.imshow(car_image, cmap='gray')
    plt.title('Example Car Image')
    plt.subplot(122)
    plt.imshow(hog_image, cmap='gray')
    plt.title('HOG Visualization')
    plt.savefig('output_images/hog.png')
    plt.show()

# will load the model from previously saved pickle if False
training = True

if training:
    # extract features and train an svc model
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

    with open('modelr.pickle', 'wb') as handle:
        pickle.dump(svc,handle)
    with open('scalerr.pickle', 'wb') as handle:
        pickle.dump(X_scaler,handle)
else:
    with open('modelr.pickle', 'rb') as handle:
        svc = pickle.load(handle)
    with open('scalerr.pickle', 'rb') as handle:
        X_scaler = pickle.load(handle)

# will hold the last n heatmaps.  The sum of these heatmaps will be used for
# thresholding in order to reduce false negatives and obtain better bounding box
heat_recent = []
# image = mpimg.imread('test_images/test1.jpg')
# out_img = process_video(image)
# plt.imshow(out_img)
# plt.show()
# out_img = cv2.cvtColor(out_img, cv2.COLOR_RGB2BGR)
# cv2.imwrite('output_images/test1_out.jpeg', out_img)

output_video = 'detect1.mp4'
clip1 = VideoFileClip("project_video.mp4")
processed_clip = clip1.fl_image(process_video)
processed_clip.write_videofile(output_video, audio=False)
