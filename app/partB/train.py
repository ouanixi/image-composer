from sklearn.neural_network import MLPClassifier
from app.utils.utils import get_numpy_data
from sklearn import svm
from sklearn.linear_model import LogisticRegression
import graphlab
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import *
from sklearn.externals import joblib


train_images = graphlab.SFrame.read_csv('features/hog_only_train.csv')
test_images = graphlab.SFrame.read_csv('features/hog_only_test.csv')
features = train_images.column_names()
features.remove('class')

X_test, Y_test = get_numpy_data(test_images, features, 'class')
X,Y = get_numpy_data(train_images, features, 'class')

# Neural network experiment
clf1 = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5,2), random_state=1)
clf1.fit(X,Y)

# SVM experiment
clf2 = svm.LinearSVC()
clf2.fit(X, Y)

print clf2.predict(X_test)
#
# KNN experiment
clf3 = KNeighborsClassifier(n_neighbors=5)
clf3.fit(X, Y)

# Naive Bayes
clf4 = GaussianNB()
clf4.fit(X, Y)

# SVM experiment
clf5 = LogisticRegression()
clf5.fit(X, Y)


print "NN", classification_report(Y_test, clf1.predict(X_test))
print "SVM", classification_report(Y_test, clf2.predict(X_test))
print "KNN", classification_report(Y_test, clf3.predict(X_test))
print "GAUSS", classification_report(Y_test, clf4.predict(X_test))
print "Logistic", classification_report(Y_test, clf5.predict(X_test))

# joblib.dump(clf2, 'model.pkl')

