from sklearn.externals import joblib
from app.utils.feature_extractor import getHog
import cv2
import numpy as np

categories = {"1": "natural", "0": "manmade"}
clf = joblib.load('../partB/svn_model.pkl')


def predict_class(img):
    features = getHog(img)
    features = np.insert(features, 0, 1)
    features = features.reshape((1,2501)).tolist()
    cls = clf.predict(features)
    return categories[str(cls[0])]


img = cv2.imread("/home/ouanixi/Work/image-composer/dataset/sun_aekhszyyonbgreke.jpg")
print predict_class(img)