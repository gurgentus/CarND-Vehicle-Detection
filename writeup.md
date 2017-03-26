**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector.
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./output_images/random_image.png
[image2]: ./output_images/hog.png
[image3]: ./output_images/HSV1032x32.jpeg
[image3_0]: ./output_images/YCrCb1032x32.jpeg
[image3_1]: ./output_images/YCrCb1048x48y400_656.jpeg
[image3_2]: ./output_images/YCrCb1064x64y400_656Scale1_5.jpeg
[image4]: ./examples/sliding_window.jpg
[image5]: ./output_images/heatmap.png
[image5_0]: ./output_images/heatmap3.png
[image5_1]: ./output_images/heatmap4.png
[image5_2]: ./output_images/heatmap5.png
[image6]: ./output_images/test1_out.jpeg
[image7]: ./output_images/test2_out.jpeg
[image8]: ./output_images/test3_out.jpeg
[image9]: ./output_images/test4_out.jpeg
[video1]: ./detect.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

This is it!

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

A wrapper function `get_hog_features()` for calling `skimage.hog()` function is located in lines 45-67.  It provides a couple of additional options having to do with visualization.  In particular if vis=True the hog features are visualized using `hog()` visualize flag.  In this case the resulting visualization image: hog_image is returned along with the features vector in an unflattened format.

The `extract_features()` function that calls `get_hog_features()` to extract features (including HOG features) from an image is located in lines 89-141 of `project.py`. In particular after optionally extracting color and histogram features, the hog features are extracted in lines 125-138

This is applied to each training image in lines 389-400.

 These images are first read and added to the `car` and `non-cars` arrays in lines 302-338.  Here is one example:

![alt text][image1]

Here is the visualization of HOG features:

![alt text][image2]




#### 2. Explain how you settled on your final choice of HOG parameters.


Next, I specify the various parameters used to extract features from the resulting images and later to use the sliding windows to classify parts of the image as car or not car.  These parameters are specified in lines 343-352.  They were picked largely by experimentation with testing on test images. In particular, choosing various color spaces and orientations parameters had the largest effect on the results.  Here is the example of trying out various parameter combinations and the resulting detections on a test image:

HSV colorspace, 10 orientations, 32x32 spatial size

![alt text][image3]

YCrCb colorspace, 10 orientations, 32x32 spatial size

![alt text][image3_0]

YCrCb colorspace, 10 orientations, 48x48 spatial size

![alt text][image3_1]

YCrCb colorspace, 10 orientations, 64x64 spatial size, y range limits

![alt text][image3_2]

In the end I chose the last configuration as the best one.

I separated my pipeline into the training part and vehicle detection part, so that when I was happy with the accuracy of the training I could just load the classifier with pickle and work on detection pipeline.  Hence, training was done only if the `training` flag was set to `True` in line 385.  

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

In line 420-424, I trained a linear SVM with the final parameter values of:

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

The sliding window search is implemented in `find_cars()` function in lines 144-206, which is based on the lectures. I chose this method as it extracts the features once for the whole frame and subsamples them later. In particular, based on specified range of y values the function takes a `spatial_size` cutouts of the image, resizes to the size of training data and runs the prediction of the classifier to check if it represents a car.  In this case the coordinates of that cutout window are appended to `bbox_list`. The function is called in lines 250-25 with two different scales.  I use a smaller scale of 1.2 for y values farther away and a larger scale of 1.8 for closer y values.  The exact parameters were obtained by experimentation.  The original overlap from lectures worked well so I left it as is.  Here is the result on a single test image.



![alt text][image3_2]

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

I experimented with various scales including using three and four different scales.  To optimize the performance I ended up using only two scales.  I also specified y ranges so that the different scales are used on different part of the images.  Here are some test image results with corresponding single frame heat maps:

![alt text][image6]

![alt text][image7]

![alt text][image8]

![alt text][image9]
---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./detect.mp4)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I created a heatmap of positive detections in each frame of the video as well as used a three dimensional numpy array to keep track of the sum of the the last 8 heatmaps.  This is done in lines 260-267.  This proved to be a simple and effective way to deal with false negatives as I could then just threshold on the sum of the heatmaps in line 275 to identify vehicle positions.  

This worked very well and removed almost all false negatives.

Finally, `scipy.ndimage.measurements.label()` was used to identify individual blobs in the aggregated heatmap and I constructed bounding boxes to cover the area of each blob detected.  

Here is an example on three of the frames with increasing total intensity of the heat map and the car identified with a resulting ounding box.

![alt text][image5_0]

![alt text][image5_1]

![alt text][image5_2]



---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  

The first task was to put together a working pipeline that would detect (even badly) cars vs no cars.  Here I mainly used the functions from the lessons.  After successfully training on my laptop I was getting some results.

The big problem I was having was with very few detections. At this point I was ready to experiment with the parameters.  I used the AWS to do a bunch of training and testing on a GPU. After experimenting with a lot of variations in the parameters, it looked like increasing HOG orientations was the biggest improvement.

The other strategy was to first concentrate on a good detection pipeline by doing a lot of experimentation.  Once that pipeline worked well on a test image, the model was saved with pickle and I concentrated on improving the search window pipeline without having to retrain every time.

One place where pipeline could fail would be with cars in the opposite traffic giving false negatives.  One way to avoid that would be to use confidence scores and associated lower scores with the cars farther away.

The other obvious place for improvement is the processing time.  Skipping some frames would reduce this.
