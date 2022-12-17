import os
import pandas as pd
import cv2
import numpy as np

data_path = '../data/'


def get_dataset(image_size):
    train_images_path, test_images_path, train_csv_path, test_csv_path = get_images_paths()


def get_images_paths():
    train_images_path = []
    test_images_path = []
    train_csv_path = []
    test_csv_path = []

    for class_folder in os.listdir(data_path):
        for sub_folder in os.listdir(data_path + class_folder):
            for img in os.listdir(data_path + class_folder + '/' + sub_folder):

                image_path = os.path.join(data_path + class_folder + '/' + sub_folder, img)

                if sub_folder == 'Train':
                    if image_path.endswith(".csv"):
                        train_csv_path.append(image_path)
                    else:
                        train_images_path.append(image_path)
                else:
                    if image_path.endswith(".csv"):
                        test_csv_path.append(image_path)
                    else:
                        test_images_path.append(image_path)

    return train_images_path, test_images_path, train_csv_path, test_csv_path


def read_images(images_paths, image_size):
    images = []

    for i in images_paths:
        image = cv2.imread(i, 0)
        image = resize_image(image, image_size)

        # image_label = create_label(i)
        images.append([np.array(image), _])


def resize_image(image, image_size):
    return cv2.resize(image, (image_size, image_size))


def create_label(image_path):
    print('label')
