import os
import pandas as pd
import cv2
import numpy as np

data_path = '../data/'


def get_dataset():
    train_images_path, test_images_path, train_csv_path, test_csv_path = get_images_paths()
    train_images = read_images(train_images_path, train_csv_path)
    test_images = read_images(test_images_path, test_csv_path)

    return train_images, test_images


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


def read_images(images_paths, csv_paths):
    csv_files = read_csv(csv_paths)

    images = []
    number_of_images = len(images_paths)

    for i in range(number_of_images):
        image = cv2.imread(images_paths[i], 0)
        image = resize_image(image, 50)

        image_name = get_image_name(images_paths[i])
        image_label = create_label(image_name, csv_files)
        images.append([np.array(image), image_label, image_name])

    return images


def resize_image(image, image_size):
    return cv2.resize(image, (image_size, image_size))


def get_image_name(image_path):
    return image_path.split('\\')[1]


def create_label(image_name, csv_files):
    image_classes = ['A', 'B', 'C', 'D', 'E']

    for i in range(len(image_classes)):
        if image_name.__contains__(image_classes[i]):
            label = csv_files[i][csv_files[i]['image_name'] == image_name]['label'].tolist()
            return label[0]


def read_csv(csv_paths):
    csv_files = []

    for i in csv_paths:
        dataFrame = pd.read_csv(i)
        csv_files.append(dataFrame)
    return csv_files
