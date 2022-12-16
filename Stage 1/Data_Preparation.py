import os
import cv2
import numpy as np
from random import shuffle
from skimage.feature import hog

data_path = '../data/'


def get_dataset(model_type, image_size):
    train_data_path, test_data_path = get_images_paths()

    train_data = read_images(train_data_path, model_type, image_size)
    test_data = read_images(test_data_path, model_type, image_size)

    shuffle(train_data)

    if model_type == 'CNN':
        X_train, Y_train = reformat_dataset(train_data, image_size)
        X_test, Y_test = reformat_dataset(test_data, image_size)
    else:
        X_train, Y_train = generate_hog_features(train_data)
        X_test, Y_test = generate_hog_features(test_data)

    return X_train, Y_train, X_test, Y_test


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


def read_images(images_paths, model_type, image_size):
    images = []

    for i in images_paths:
        image = cv2.imread(i, 0)
        image = resize_image(image, image_size, model_type)

        image_label = create_label(i)
        images.append([np.array(image), image_label])

    return images


def resize_image(image, image_size, model_type):
    if model_type == 'HOG':
        return cv2.resize(image, (image_size, 2 * image_size))
    else:
        return cv2.resize(image, (image_size, image_size))


def create_label(image_path):
    image_label = image_path.split('/')[1]
    image_classes = ['A', 'B', 'C', 'D', 'E']
    label_encoded = np.zeros((5, 1))

    for i in range(len(image_classes)):
        if image_label.endswith(image_classes[i]):
            label_encoded[i] = 1

    return label_encoded


def reformat_dataset(data, image_size):
    X = np.array([i[0] for i in data], dtype=object).reshape(-1, image_size, image_size, 1)
    Y = np.array([i[1] for i in data])
    Y = Y.reshape(len(Y), 5)

    return X, Y


def generate_hog_features(data):
    hog_features = []
    images_labels = []

    for i in data:
        fd = hog(i[0], orientations=9, pixels_per_cell=(8, 8),
                 cells_per_block=(2, 2), visualize=False, multichannel=False)

        hog_features.append(fd)
        images_labels.append(i[1])

    return np.array(hog_features), np.array(images_labels)
