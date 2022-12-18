import Data_Preparation
import cv2
import numpy as np
from sklearn.cluster import KMeans


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
    kmeans_object = KMeans(n_clusters=100)
    return kmeans_object.fit_predict(descriptors_stack)


kps, desc = extract_sift_features(train_images)

desc = transform_list(desc)

kmeansreturn = cluster(desc)

print(type(kmeansreturn))
print(len(kmeansreturn))
print(kmeansreturn)
print(kmeansreturn.shape)