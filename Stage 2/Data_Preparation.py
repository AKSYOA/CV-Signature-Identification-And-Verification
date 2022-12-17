import os
import pandas as pd
import cv2


data_path = '../data/'


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


path1, path2, csv1, csv2 = get_images_paths()
