import Data_Preparation
import cv2
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

train_images, test_images = Data_Preparation.get_dataset()


def extract_sift_features(data):
    sift_object = cv2.SIFT_create()

    key_points = []
    Descriptors = []

    for i in range(len(data)):
        kp, desc = sift_object.detectAndCompute(data[i][0], None)
        key_points.append(kp)
        Descriptors.append(desc)

    return key_points, Descriptors


def transform_list(l):
    vStack = np.array(l[0])

    for i in range(len(l)):
        if i == 0:
            continue
        vStack = np.vstack((vStack, l[i]))

    return vStack


def cluster(descriptors_stack):
    kmeans_object = KMeans(n_clusters=20)
    return kmeans_object.fit_predict(descriptors_stack)


def developVocabulary(n_images, descriptor_list, k_means_result):
    mega_histogram = np.array([np.zeros(100) for i in range(n_images)])
    count = 0
    for i in range(n_images):
        l = len(descriptor_list[i])
        for j in range(l):
            idx = k_means_result[count + j]
            mega_histogram[i][idx] += 1
        count += l
    return mega_histogram


def standardize(mega_histogram):
    scale = StandardScaler().fit(mega_histogram)
    return scale.transform(mega_histogram)


def train_model(mega_histogram, train_labels):
    clf = SVC()
    clf.fit(self.mega_histogram, train_labels)
    return clf
