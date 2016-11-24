from sklearn.externals import joblib
from app.utils.feature_extractor import getHog
import cv2
import numpy as np
import graphlab
from app.utils.utils import get_numpy_data

categories = {"1": "natural", "0": "manmade"}
clf = joblib.load('../partB/model.pkl')

test_images = graphlab.SFrame.read_csv('hog_only_test.csv')
features = test_images.column_names()
features.remove('class')

X_test, Y_test = get_numpy_data(test_images, features, 'class')


def predict_class(img):
    features = getHog(img)
    features = np.insert(features, 0, 1)
    features = np.vstack((X_test, features.reshape(1,2501)))
    cls = clf.predict(features)
    return cls

#
img = cv2.imread("/home/ouanixi/Work/image-composer/dataset/sun_aabghtsyctpcjvlc.jpg")
print predict_class(img)