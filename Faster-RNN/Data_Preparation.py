import os
import cv2
import numpy as np
import csv

data_path = 'SignatureObjectDetection/'


def get_data_csvfile():
    train_path, train_dimensions_path, test_path, test_dimensions_path = get_images_paths()

    make_data_as_csv(train_dimensions_path[0],'train')
    make_data_as_csv(test_dimensions_path[0],'test')

    return train_path[0][0],test_path[0][0]

def get_images_paths():
    train_path = []
    test_path = []
    train_dimensions_path = []
    test_dimensions_path = []

    for i in os.listdir(data_path):
        if i == 'TrainImages':
            for img in os.listdir(os.path.join(data_path, i)):
                train_path.append([os.path.join(data_path, i), img])  # (link ,img_name)
        elif i == "TestImages":
            for img in os.listdir(os.path.join(data_path, i)):
                test_path.append([os.path.join(data_path, i), img])
        elif i == "TrainGroundTruth":
            train_dimensions_path.append(os.path.join(data_path, i))
        elif i == "TestGroundTruth":
            test_dimensions_path.append(os.path.join(data_path, i))
    return train_path, train_dimensions_path, test_path, test_dimensions_path



def make_data_as_csv(path, mode):
    if (mode == 'train'):
        file_name = 'train_data.csv'
    elif mode == 'test':
        file_name = 'test_data.csv'

    with open(file_name, 'w', newline='') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter=',',
                                quotechar='|', quoting=csv.QUOTE_MINIMAL)
        for img_name_txt in os.listdir(path):
            text_file = open(os.path.join(path, img_name_txt), "r")
            Lines = text_file.readlines()
            for j in Lines:
                line_split = j.strip().split(',')
                img = img_name_txt[:-3] + 'tif'
                spamwriter.writerow([img, 'Signature', line_split[0], line_split[1], line_split[2], line_split[3]])

