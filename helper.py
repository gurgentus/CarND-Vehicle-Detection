import math
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2

def grayscale(img):
    """Applies the Grayscale transform
    This will return an image with only one color channel
    but NOTE: to see the returned image as grayscale
    you should call plt.imshow(gray, cmap='gray')"""
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Or use BGR2GRAY if you read an image with cv2.imread()
    # return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def canny(img, low_threshold, high_threshold):
    """Applies the Canny transform"""
    return cv2.Canny(img, low_threshold, high_threshold)

def gaussian_blur(img, kernel_size):
    """Applies a Gaussian Noise kernel"""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

def region_of_interest(img, vertices):
    """
    Applies an image mask.

    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    """
    #defining a blank mask to start with
    mask = np.zeros_like(img)

    #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

    #filling pixels inside the polygon defined by "vertices" with the fill color
    cv2.fillPoly(mask, vertices, ignore_mask_color)

    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


def draw_lines(img, lines, numColorLines, color=[255, 0, 0], thickness=15):
    """
    NOTE: this is the function you might want to use as a starting point once you want to
    average/extrapolate the line segments you detect to map out the full
    extent of the lane (going from the result shown in raw-lines-example.mp4
    to that shown in P1_example.mp4).

    Think about things like separating line segments by their
    slope ((y2-y1)/(x2-x1)) to decide which segments are part of the left
    line vs. the right line.  Then, you can average the position of each of
    the lines and extrapolate to the top and bottom of the lane.

    This function draws `lines` with `color` and `thickness`.
    Lines are drawn on the image inplace (mutates the image).
    If you want to make the lines semi-transparent, think about combining
    this function with the weighted_img() function below
    """

    # img contains the image onto which to draw the lanes
    # lines are the lines making up a potential lane
    # first 'numColorLines' of the lines array have the lines generated from the color information (white and yellow)
    # these will carry higher weight in the processing

    imshape = img.shape

    # will keep track of potential candidates for a lane in Hough Space
    # actual theta ranges in [-pi/2, pi/2],
    # in processing will add pi/2 and keep track of theta in degrees as index between [0, thbound]
    # actual r ranges in [-rbound, rbound], in processing will keep track as index between [0, 2*rbound+1]
    thbound = 181
    rbound = np.maximum(imshape[0],imshape[1])

    # will keep track of key attributes of potential lines, slope, y-intercept, and x-intercept
    slope = np.zeros(shape=(thbound, 2*rbound+1))
    yint = np.zeros(shape=(thbound, 2*rbound+1))
    xint = np.zeros(shape=(thbound, 2*rbound+1))

    # also keep track of how many times a close attribute is repeated, this will increase confidence
    # in the prediction
    numDataPoints = np.zeros(shape=(thbound, 2*rbound+1))

    # keep track of confidence level in the line prediction
    # lines will be recognized as part of a lane if the confidence is above confThreshold
    # several factors will determine the confidence:
    # - line being close to white or yellow
    # - many lines with the same slope and y-int
    # - theta close to value at which we expect to see a lane
    conf = np.ones(shape=(thbound, 2*rbound+1))
    confThreshold = 100

    # start processing lines
    # to avoid errors will process almost horizontal and almost vertical lines separately
    count = 0
    weight = 1
    if lines is not None:
        for line in lines:
            for x1,y1,x2,y2 in line:
                if (np.abs(y2-y1) < 0.01):
                    m = 0
                    b = y1
                    th = 90
                    xi = np.Inf
                    r = b
                elif (np.abs(x2-x1) < 0.01):
                    m = np.Inf
                    b = np.Inf
                    th = 0
                    xi = x1
                    r = np.Inf
                else:
                    m = (y2-y1)/(x2-x1)
                    b = -m*x1+y1
                    xi = (imshape[0]-b)/m
                    th = int(round((np.arctan(-1/m)*(180/np.pi))))
                    r = b*np.sin(th)

                if not np.isinf(r) and r > -rbound and r < rbound:
                    # if theta is within range update characteristics of the line and the confidence level
                    # based on the weight
                    # if the line was obtained based on yellow or white color update the weight which will
                    # be multiplied by the length, i.e. long white or yellow lines carry a lot of weight

                    th = th + 90
                    r = int(round(r))+rbound
                    weight = 1

                    if count < numColorLines:
                        weight = weight + 1

                    # update the weight based on the angle, the more 'flat' the line the less likely it is a lane
                    weight = weight + (th - 30)*(150-th)/1000

                    # if outside of visible area, put weight = 0
                    if (xi >= imshape[1]):
                        weight = 0
                    if (xi < 0):
                        weight = 0

                    # update confidence and line attributes based on the latest processed line
                    conf[th][r] = conf[th][r] + weight*((x2-x1)*(x2-x1)+(y2-y1)*(y2-y1))
                    slope[th][r] = slope[th][r] + m
                    yint[th][r] = yint[th][r] + b
                    xint[th][r] = xint[th][r] + xi
                    numDataPoints[th][r] = numDataPoints[th][r] + 1
                count = count + 1

        # filter lines with low confidence threshold
        relevantAnglesUns = zip(*np.where(conf >= confThreshold))


        # average close lines
        # sort in increasing angle order and loop through the angles averaging out nearby angles and r values
        # for angles a weighted average is calculated so that low confidence angles have less weight
        relevantAngles = sorted(relevantAnglesUns, key=lambda x: x[0])
        # keep track of previous angle and r: pind1 and pind2
        pind1 = -10
        pind2 = -10
        anCount = 1
        angleSum = 0
        yintSum = 0
        slopeSum = 0
        count = 0
        finishedProcessing = False
        for ind1,ind2 in relevantAngles:
                yint[ind1][ind2] = yint[ind1][ind2]/numDataPoints[ind1][ind2]
                slope[ind1][ind2] = slope[ind1][ind2]/numDataPoints[ind1][ind2]
                count = count + 1
                # if first run through, initialize the first angle weighted by confidence
                if (count == 1):
                    angleSum = ind1*conf[ind1][ind2]
                    yintSum = yint[ind1][ind2]*conf[ind1][ind2]
                    slopeSum = slope[ind1][ind2]*conf[ind1][ind2]
                    finishedProcessing = True
                    pind1 = ind1
                    pind2 = ind2
                # otherwise check if within range of the previous angle and update weighted angleSum and other
                # attributes, if not within range will finish processing by renormalizing the attributes
                else:
                    if (np.abs(ind1-pind1) < 3) and (np.abs(ind2-pind2) < 1000):
                        anCount = anCount + 1
                        numDataPoints[pind1][pind2] = numDataPoints[pind1][pind2] + numDataPoints[ind1][ind2]
                        angleSum = angleSum + ind1*conf[ind1][ind2]
                        slopeSum = slopeSum + slope[ind1][ind2]*conf[ind1][ind2]
                        yintSum = yintSum + yint[ind1][ind2]*conf[ind1][ind2]

                        xint[pind1][pind2] = xint[pind1][pind2] + xint[ind1][ind2]
                        conf[pind1][pind2] = conf[pind1][pind2] + conf[ind1][ind2]
                        conf[ind1][ind2] = 0
                    else:
                        finishedProcessing = False

                if (count == len(relevantAngles)) or (not finishedProcessing):
                        avgAngle = int(round(angleSum/conf[pind1][pind2]))
                        slope[avgAngle][pind2] = slopeSum/conf[pind1][pind2]
                        numDataPoints[avgAngle][pind2] = numDataPoints[pind1][pind2]
                        yint[avgAngle][pind2] = yintSum/conf[pind1][pind2]
                        xint[avgAngle][pind2] = xint[pind1][pind2]/numDataPoints[pind1][pind2]
                        conf[avgAngle][pind2] = conf[pind1][pind2]
                        if (avgAngle != pind1):
                            conf[pind1,pind2] = 0
                        finishedProcessing = True
                        pind1 = ind1
                        pind2 = ind2
                        angleSum = ind1*conf[ind1][ind2]
                        yintSum = yint[ind1][ind2]*conf[ind1][ind2]
                        slopeSum = slope[ind1][ind2]*conf[ind1][ind2]

        # get the updated relevant angles based on the averaging in the previous step
        relevantAngles = zip(*np.where(conf >= confThreshold))

        # now get the two different facing lines with highest confidence value
        angle1 = 181
        intercept1 = -1
        angle2 = -1
        intercept2 = -1
        conf1 = -1
        conf2 = -1

        for ind1,ind2 in relevantAngles:
            #print(ind1, ind2, conf[ind1][ind2] )
            if (ind1 <= 90) and (conf[ind1][ind2] > conf1):
                angle1 = ind1
                intercept1 = ind2
                conf1 = conf[ind1][ind2]
            if (ind1 >= 90) and (conf[ind1][ind2] > conf2):
                angle2 = ind1
                intercept2 = ind2
                conf2 = conf[ind1][ind2]
        relevantAngles = zip(*np.where(conf >= confThreshold))

        # finally draw the two lines
        pts = []
        for ind1,ind2 in relevantAngles:
            if (ind1 == angle1 and ind2 == intercept1) or (ind1 == angle2 and ind2 == intercept2):
                m = slope[ind1][ind2]
                b = yint[ind1][ind2]
                if not np.isinf(m) and not np.isinf(b):
                    if np.abs(m) > 0.01:
                        pts.append([int(round((imshape[0]-b)/m)), imshape[0]])
                        pts.append([int(round((0.7*imshape[0]-b)/m)), int(round(0.7*imshape[0]))])
        if len(pts) == 4:
            cv2.fillConvexPoly(img, cv2.convexHull(np.int_(pts)), (0,255, 0))

def hough_lines(img, colormask, rho, theta, threshold, min_line_len, max_line_gap):
    """
    `img` should be the output of a Canny transform.

    Returns an image with hough lines drawn.
    """
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    colorLines = cv2.HoughLinesP(colormask, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    np.append(colorLines,lines)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    if colorLines is not None:
        draw_lines(line_img, lines, len(colorLines))
    else:
        draw_lines(line_img, lines, 0)
    return line_img

# Python 3 has support for cool math symbols.

def weighted_img(img, initial_img, α=0.8, β=1., λ=0.):
    """
    `img` is the output of the hough_lines(), An image with lines drawn on it.
    Should be a blank image (all black) with lines drawn on it.

    `initial_img` should be the image before any processing.

    The result image is computed as follows:

    initial_img * α + img * β + λ
    NOTE: initial_img and img must be the same shape!
    """
    return cv2.addWeighted(initial_img, α, img, β, λ)

def process_img(image):
    # define a rough region where we expect the lines
    imshape = image.shape
    # select the region from about 2/3 of the full height, this should be the most reliable region
    # to make it more robust do not assume anything else about the region
    vertices = np.array([[(200,2*imshape[0]/2),(0,imshape[0]), (imshape[1],imshape[0]), (imshape[1]-200, 2*imshape[0]/2)]], dtype=np.int32)

    # will pick edges using both color and grayscale
    # the code in hough_lines will use both edges with higher weight given to lines identified through color
    color_threshold = 150
    imageHSV = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    yellowMask = cv2.inRange(imageHSV, np.array([10, 0, 0]), np.array([50, 255, 255]))

    whiteMask = cv2.inRange(image, np.array([200, 200, 200]), np.array([255,255,255]))
    mask = yellowMask | whiteMask
    updated_color_region = region_of_interest(mask, vertices)

    # Smooth and detect edges obtained from color image using Canny
    kernel_size = 5
    updated_color_region = gaussian_blur(updated_color_region, kernel_size)
    low_threshold = 30
    high_threshold = 60
    edgesC = canny(updated_color_region, low_threshold, high_threshold)


    # Smooth and detect edges obtained from grayscale using Canny
    gray = grayscale(image)
    kernel_size = 5
    blur_gray = gaussian_blur(gray, kernel_size)
    low_threshold = 50
    high_threshold = 100
    edges = canny(blur_gray, low_threshold, high_threshold)

    # select the region of interest
    updated_region = region_of_interest(edges, vertices)
    updated_color_region = region_of_interest(edgesC, vertices)

    # Define the Hough transform parameters
    rho = 1 # distance resolution in pixels of the Hough grid
    theta = np.pi/180 # angular resolution in radians of the Hough grid
    threshold = 30     # minimum number of votes (intersections in Hough grid cell)
    min_line_len = 7 #minimum number of pixels making up a line
    max_line_gap = 3   # maximum gap in pixels between connectable line segments
    line_image = np.copy(image)*0 # creating a blank to draw lines on

    # Run Hough on edge detected images, here the function takes both grayscale
    # and color image and will combine the obtained lines
    image_with_lines = hough_lines(updated_region, updated_color_region, rho, theta, threshold, min_line_len, max_line_gap)

    final_image = weighted_img(image_with_lines, image, α=1, β=0.3, λ=0.)

    return final_image
