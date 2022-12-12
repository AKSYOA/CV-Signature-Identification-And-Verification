import cv2
import numpy as np
import os
import Data_Preparation
from sklearn.svm import SVC

sift = cv2.SIFT_create()
X_train, Y_train, X_test, Y_test = Data_Preparation.get_dataset()
model =SVC()

def getSift():
    list_kp =[]
    for img in X_train:
        kp, desc = sift.detectAndCompute(img, None)
        img = cv2.drawKeypoints(img, kp, None)
        list_kp.append(desc)
    return list_kp





