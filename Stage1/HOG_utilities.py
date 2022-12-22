from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import numpy as np
from skimage.feature import hog
from joblib import dump, load
import os


def generate_HOG_model(train_data):
    X_train, Y_train = generate_hog_features(train_data)

    if os.path.exists('../Trained Models/HOG_model.joblib'):
        clf = load('../Trained Models/HOG_model.joblib')
    else:
        clf = LogisticRegression().fit(X_train, Y_train)
        dump(clf, '../Trained Models/HOG_model.joblib')

    return clf


def generate_hog_features(data):
    hog_features = []
    images_labels = []

    for i in data:
        fd = hog(i[0], orientations=9, pixels_per_cell=(8, 8),
                 cells_per_block=(2, 2), visualize=False, multichannel=False)

        hog_features.append(fd)
        images_labels.append(reformat_label(i[1]))

    return np.array(hog_features), np.array(images_labels)


def reformat_label(image_label):
    return np.argmax(image_label)


def testModel(test_data, clf):
    X_test, Y_test = generate_hog_features(test_data)
    predictions = clf.predict(X_test)

    print("Testing Accuracy: " + str(accuracy_score(Y_test, predictions)))

    matrix = confusion_matrix(Y_test, predictions)

    disp = ConfusionMatrixDisplay(confusion_matrix=matrix)
    disp.plot()
    plt.show()
