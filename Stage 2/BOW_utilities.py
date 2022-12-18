import Data_Preparation
import cv2
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

train_images, test_images = Data_Preparation.get_dataset()


def generate_BOW_model(data):
    kps, descriptors_list = extract_sift_features(data)

    descriptors_stack = transform_list(descriptors_list)

    k_means_result = cluster(descriptors_stack)

    mega_histogram = developVocabulary(len(data), descriptors_list, k_means_result)
    print(mega_histogram.shape)

    mega_histogram = standardize(mega_histogram)

    data_labels = get_labels(data)
    model = train_model(mega_histogram, data_labels)

    return model


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
    k_means_object = KMeans(n_clusters=20)
    return k_means_object.fit_predict(descriptors_stack)


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
    clf.fit(mega_histogram, train_labels)
    return clf


def get_labels(data):
    labels = []

    for i in range(len(data)):
        labels.append(data[i][1])

    return np.array(labels)
