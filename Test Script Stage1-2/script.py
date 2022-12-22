from joblib import load
from tensorflow import keras

import Stage1.Data_Preparation
import Stage1.HOG_utilities
from Stage1.CNN_utilities import buildSequentialModel
import numpy as np

personClasses = ['personA', 'personB', 'personC', 'personD', 'personE']


def Identify(image, model_type):
    if model_type == 1:
        image = Stage1.Data_Preparation.read_images(image, model_type='HOG', image_size=64)
        image, _ = Stage1.HOG_utilities.generate_hog_features(image)
        clf = load('../Trained Models/HOG_model.joblib')
        imageClass = clf.predict(image)
        return personClasses[imageClass[0]]
    else:
        image = Stage1.Data_Preparation.read_images(image, model_type='CNN', image_size=128)
        image, _ = Stage1.CNN_utilities.reformat_dataset(image, image_size=128)
        model = buildSequentialModel()
        model = keras.models.load_model('../Trained Models/basic_CNN_Model.h5')
        imageClass = model.predict(image)
        return personClasses[np.argmax(imageClass)]
