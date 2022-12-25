import cv2
from joblib import load
from tensorflow import keras
import Stage1.Data_Preparation
import Stage1.HOG_utilities
import Stage2.Data_Preparation
import Stage2.Siamese_utilities
import Stage2.BOW_utilities
import Stage1.CNN_utilities
import numpy as np
import random

person_classes = ['personA', 'personB', 'personC', 'personD', 'personE']
verification_classes = ['forged', 'real']


def Identify(image_path, model_type):
    if model_type == 1:
        image = Stage1.Data_Preparation.read_images(image_path, model_type='HOG', image_size=64)
        image, _ = Stage1.HOG_utilities.generate_hog_features(image)
        clf = load('../Trained Models/HOG_model.joblib')
        imageClass = clf.predict(image)
        return person_classes[imageClass[0]], imageClass[0]
    else:
        image = Stage1.Data_Preparation.read_images(image_path, model_type='CNN', image_size=128)
        image, _ = Stage1.CNN_utilities.reformat_dataset(image, image_size=128)
        model = Stage1.CNN_utilities.buildSequentialModel()
        model = keras.models.load_model('../Trained Models/basic_CNN_Model.h5')
        imageClass = model.predict(image)
        return person_classes[np.argmax(imageClass)], np.argmax(imageClass)


def Verify(image_path, model_type, person_class_index):
    if model_type == 2:
        prediction = prepare_Siamese_model(image_path, person_class_index)
        return verification_classes[not prediction[0]]
    else:
        image = read_image(image_path)
        image = np.expand_dims(image, axis=1)
        model_path = "../Trained Models/BOW_{person}_model.joblib".format(person=person_classes[person_class_index])
        k_means_path = "../Trained Models/BOW_{person}_k_means.joblib".format(person=person_classes[person_class_index])
        model = load(model_path)
        k_means_object = load(k_means_path)
        prediction = Stage2.BOW_utilities.test_model(image, model, k_means_object, 300)
        return verification_classes[prediction[0][0]]


def prepare_Siamese_model(image_path, person_class_index):
    train_images, _ = Stage2.Data_Preparation.get_dataset()
    train_images.sort(key=lambda a: a[2])
    train_images.sort(key=lambda a: a[1])

    image = read_image(image_path)

    start = person_class_index * 20
    end = (person_class_index + 1) * 20
    r1 = random.randint(start + 100, end + 100)

    positive_image = np.array([train_images[r1][0]])  # real

    encoder = Stage2.Siamese_utilities.get_encoder((128, 128, 3))
    encoder.load_weights('../Trained Models/encoder.h5')

    tensor1 = encoder.predict(image)
    tensor2 = encoder.predict(positive_image)

    return classify_images(tensor1, tensor2)


def classify_images(tensor1, tensor2, threshold=1.3):
    distance = np.sum(np.square(tensor1 - tensor2), axis=-1)
    prediction = np.where(distance <= threshold, 0, 1)
    return prediction


def read_image(image_path):
    image = cv2.imread(image_path[0])
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (128, 128))
    image = [image]
    return np.array(image)
