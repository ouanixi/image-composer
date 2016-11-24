import cv2
import HistogramHandler
from skimage.feature import hog
from skimage import color, exposure
import numpy as np


def getHistogram(image):
   hist = cv2.calcHist([image], [0, 1, 2], None, [8, 8, 8],
                       [0, 256, 0, 256, 0, 256])

   hist = hist.flatten()
   return hist


def harris_corner_detection(img):
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    img = cv2.resize(img, (25, 25), interpolation=cv2.INTER_AREA)
    img = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = np.float32(gray)

    dst = cv2.cornerHarris(gray,2,3,0.04)
    dst = cv2.dilate(dst,None)

    img[dst>0.01*dst.max()]=[0,0,255]

    return img.flatten()

def getAverageColor(image, index, bins):
    (h, w, _) = image.shape
    histogram = cv2.calcHist([image], [index], None, [bins], [0, bins])
    x = 0
    for i in range(0, len(histogram)):
        x += (int(histogram[i]) * i)
    return x / (w * h)


def getEdgeImage(img):
    edges = cv2.resize(img, (25, 25), interpolation=cv2.INTER_AREA)
    edges = cv2.fastNlMeansDenoisingColored(edges, None, 10, 10, 7, 21)
    edges = cv2.Canny(edges, 180, 200)
    return edges.flatten()


# http://docs.opencv.org/3.0-beta/doc/py_tutorials/py_imgproc/py_houghlines/py_houghlines.html For Hough Transform. Accessed: 14/11/2016
def getHoughTransformLines(img):
    length = 0
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    minLineLength = 100
    maxLineGap = 10
    lines = cv2.HoughLinesP(img_edges, 1, np.pi / 180, 100, minLineLength, maxLineGap)

    if lines is not None:
        length = len(lines)

    return [length]


def getHog(img):
    img = color.rgb2gray(img)
    img = cv2.resize(img, (50, 50), interpolation=cv2.INTER_AREA)
    fd, hog_image = hog(img, orientations=8, pixels_per_cell=(16, 16),
                        cells_per_block=(1, 1), visualise=True)
    return hog_image.flatten()


def combine_hog_lines(img):
    features = np.append(getHog(img), getHoughTransformLines(img))
    return features


def combine_hog_corners(img):
    features = np.append(getHog(img), harris_corner_detection(img))
    return features


def extractFeature(image):
    entry = {}
    entry["b"] = getAverageColor(image, 0, 256)
    entry["g"] = getAverageColor(image, 1, 256)
    entry["r"] = getAverageColor(image, 2, 256)
    entry["histogram"] = HistogramHandler.calc_histogram(image)
    return entry

