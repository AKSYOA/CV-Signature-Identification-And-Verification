import os
import cv2
import numpy as np

data_path = 'SignatureObjectDetection/'


def get_dataset(image_size=150):
    train_path, train_dimensions_path, test_path, test_dimensions_path = get_images_paths()
    train_images = read_images(train_path, image_size)
    test_images = read_images(test_path, image_size)
    train_boxes  = get_boxes_dimensions(train_dimensions_path)
    test_boxes = get_boxes_dimensions(test_dimensions_path)
    # print(len(test_images))
    # print(len(test_boxes))
    return train_images, train_boxes, test_images ,test_boxes


def get_images_paths():
    train_path = []
    test_path = []
    train_dimensions_path = []
    test_dimensions_path = []

    for i in os.listdir(data_path):
        if i == 'TrainImages':
            for img in os.listdir(os.path.join(data_path, i)):
                train_path.append([os.path.join(data_path, i), img]) #(link ,img_name)
        elif i == "TestImages":
            for img in os.listdir(os.path.join(data_path, i)):
                test_path.append([os.path.join(data_path, i), img])
        elif i == "TrainGroundTruth":
            train_dimensions_path.append(os.path.join(data_path, i))
        elif i == "TestGroundTruth":
            test_dimensions_path.append(os.path.join(data_path, i))
    return train_path, train_dimensions_path, test_path, test_dimensions_path


def read_images(images_paths, image_size):
    images = []

    for i in range(len(images_paths)):
        image = cv2.imread(os.path.join(images_paths[i][0], images_paths[i][1]), 0)
        image = resize_image(image, image_size)
        # cv2.imshow('Filtered_img', image)
        # cv2.waitKey(0)

        images.append([images_paths[i][1], np.array(image)])

    return images


def resize_image(image, image_size):
    return cv2.resize(image, (image_size, 2 * image_size))


def get_boxes_dimensions(path):
    dimension_images = []
    for i in os.listdir(path[0]):
        text_file = open(os.path.join(path[0], i), "r")
        Lines = text_file.readlines()
        dimension_boxes = []
        for j in Lines:
            # print(i, " ", j)
            dimension_boxes.append(j)
        dimension_images.append(dimension_boxes)
    return dimension_images
