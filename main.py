import tensorflow as tf
from tensorflow import keras
import os
import cv2

data_path = 'data/'


def get_images_paths():
    train_data_path = []
    test_data_path = []

    for class_folder in os.listdir(data_path):
        for sub_folder in os.listdir(data_path + class_folder):
            for img in os.listdir(data_path + class_folder + '/' + sub_folder):

                image_path = os.path.join(data_path + class_folder + '/' + sub_folder, img)
                if image_path.endswith(".csv"):
                    continue

                if sub_folder == 'Train':
                    train_data_path.append(image_path)
                else:
                    test_data_path.append(image_path)

    return train_data_path, test_data_path


def read_images(images_paths):
    for i in images_paths:
        image = cv2.imread(i, 0)
        cv2.imshow("image", image)
        cv2.waitKey(0)


train_data_path, test_data_path = get_images_paths()
read_images(train_data_path)
