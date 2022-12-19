from sklearn.linear_model import LogisticRegression

import cv2
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


def generate_BOW_model(data, n_clusters):
    kps, descriptors_list = extract_sift_features(data)

    descriptors_stack = transform_list(descriptors_list)

    k_means_result, k_means_object = cluster(descriptors_stack, n_clusters)

    mega_histogram = developVocabulary(len(data), descriptors_list, k_means_result, n_clusters)

    mega_histogram = standardize(mega_histogram)

    data_labels = get_labels(data)
    model = train_model(mega_histogram, data_labels)

    return model, k_means_object


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

    for remaining in l[1:]:
        vStack = np.vstack((vStack, remaining))

    return vStack


def cluster(descriptors_stack, n_clusters):
    k_means_object = KMeans(n_clusters=n_clusters)
    k_means_result = k_means_object.fit_predict(descriptors_stack)
    return k_means_result, k_means_object


def developVocabulary(n_images, descriptor_list, k_means_result, n_clusters):
    mega_histogram = np.array([np.zeros(n_clusters) for i in range(n_images)])
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
    scale_result = scale.transform(mega_histogram)
    return scale_result


def train_model(mega_histogram, train_labels):
    clf = LogisticRegression()
    clf.fit(mega_histogram, train_labels)
    return clf


def get_labels(data):
    labels = []

    for i in range(len(data)):
        labels.append(data[i][1])

    return np.array(labels)


def recognize_image(k_means_object, image_descriptors, n_clusters):
    vocabulary = np.array([[0 for i in range(n_clusters)]])
    vocabulary = np.array(vocabulary, 'float32')

    k_means_result = k_means_object.predict(image_descriptors)

    for each in k_means_result:
        vocabulary[0][each] += 1

    return vocabulary


def test_model(test_data, model, k_means_object, n_clusters):
    predictions = []

    kps, descriptors_list = extract_sift_features(test_data)
    for i in range(len(test_data)):
        vocabulary = recognize_image(k_means_object, descriptors_list[i], n_clusters)
        label = model.predict(vocabulary)
        print(label)
        predictions.append(label)

    return predictions
