import os
import cv2
import numpy as np

data_path = 'SignatureObjectDetection/'


def get_dataset(image_size=80):
    train_path, train_dimensions_path, test_path, test_dimensions_path = get_images_paths()
    train_imgs = read_images(train_path,image_size)
    test_imgs = read_images(test_path,image_size)
    train_boxs = get_boxs_dimensions(train_dimensions_path)
    test_boxs = get_boxs_dimensions(test_dimensions_path)
    # print(len(test_imgs))
    # print(len(test_boxs))

    return train_imgs, train_boxs, test_imgs, test_boxs


def get_images_paths():
    train_path = []
    test_path = []
    train_dimensions_path = []
    test_dimensions_path = []

    for i in os.listdir(data_path):
        print(i)
        if i == 'TrainImages':
            for img_path in os.listdir(os.path.join(data_path,i)):
                train_path.append(img_path)
        elif i == "TestImages":
            for img_path in os.listdir(os.path.join(data_path,i)):
                test_path.append(img_path)
        elif i == "TrainGroundTruth":
            train_dimensions_path.append(os.path.join(data_path,i))
        elif i == "TestGroundTruth":
            test_dimensions_path.append(os.path.join(data_path,i))
    return train_path, train_dimensions_path, test_path, test_dimensions_path


def read_images(images_paths, image_size):
    images = []

    for i in images_paths:
        image = cv2.imread(i, 0)
        # image = resize_image(image, image_size)

        images.append([np.array(image)])

    return images


# def resize_image(image, image_size):
#      return cv2.resize(image, (image_size, 2 * image_size))


def get_boxs_dimensions(path):
    dimension_imgs = []
    for i in os.listdir(path[0]):
        text_file = open(os.path.join(path[0], i), "r")
        Lines = text_file.readlines()
        dimension_boxs = []
        for j in Lines:
            dimension_boxs.append(j)
        dimension_imgs.append(dimension_boxs)
    return dimension_imgs


get_dataset(80)
