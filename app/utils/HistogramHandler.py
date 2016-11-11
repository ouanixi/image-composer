import json
import cv2
import math
import sys
import feature_extractor
import numpy as np


def calc_chi_squared_dist(histogram1, histogram2):
    distance = cv2.compareHist(histogram1, histogram2, method=cv2.HISTCMP_CHISQR)
    return distance


#http://stackoverflow.com/questions/1401712/how-can-the-euclidean-distance-be-calculated-with-numpy
def calc_euclidean_dist(histogram1, histogram2):
    distance = np.sqrt(np.sum((histogram1 - histogram2)**2))
    return distance


def calc_hist_intersection(histogram1, histogram2):
    intersection = cv2.compareHist(histogram1, histogram2, method=cv2.HISTCMP_INTERSECT)
    return intersection


def calc_histogram(image):
    image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    #https://github.com/opencv/opencv/blob/master/samples/python/color_histogram.py
    hist = cv2.calcHist(image_hsv, [0, 1], None, [180, 256], [0, 180, 0, 256])
    #http://stackoverflow.com/questions/9390592/drawing-histogram-in-opencv-python
    cv2.normalize(hist, hist, 0, 255, cv2.NORM_MINMAX)
    return hist;


#https://www.mathworks.com/help/stats/knnsearch.html
#^ For Part B