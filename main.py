import tensorflow as tf
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow import keras
import os
import cv2
import numpy as np

data_path = 'data/'


def get_images_paths():
    train_images_path = []
    test_images_path = []

    for class_folder in os.listdir(data_path):
        for sub_folder in os.listdir(data_path + class_folder):
            for img in os.listdir(data_path + class_folder + '/' + sub_folder):

                image_path = os.path.join(data_path + class_folder + '/' + sub_folder, img)
                if image_path.endswith(".csv"):
                    continue

                if sub_folder == 'Train':
                    train_images_path.append(image_path)
                else:
                    test_images_path.append(image_path)

    return train_images_path, test_images_path


def read_images(images_paths):
    images = []
    for i in images_paths:
        image = cv2.imread(i, 0)
        image_label = create_label(i)
        images.append([np.array(image), image_label])
    return images


def create_label(image_path):
    image_label = image_path.split('/')[1]
    image_classes = ['A', 'B', 'C', 'D', 'E']
    label_encoded = np.zeros((5, 1))
    for i in range(len(image_classes)):
        if image_label.endswith(image_classes[i]):
            label_encoded[i] = 1
    return label_encoded


train_data_path, test_data_path = get_images_paths()
train_data = read_images(train_data_path)
test_data = read_images(test_data_path)
print(len(train_data))
print(len(test_data))
