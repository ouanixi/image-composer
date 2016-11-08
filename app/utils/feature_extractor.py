import cv2
import HistogramHandler


def getAverageColor(image, index, bins):
    (h, w, _) = image.shape
    histogram = cv2.calcHist([image], [index], None, [bins], [0, bins])
    x = 0
    for i in range(0, len(histogram)):
        x += (int(histogram[i]) * i)
    return x / (w * h)

def extractFeature(image):
    entry = {}
    entry["b"] = getAverageColor(image, 0, 256)
    entry["g"] = getAverageColor(image, 1, 256)
    entry["r"] = getAverageColor(image, 2, 256)
    entry["histogram"] = HistogramHandler.calc_histogram(image).tolist()
    return entry

