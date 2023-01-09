from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from tensorflow.keras import layers
import os
import keras
import cv2

image_Classes = ['PersonA', 'PersonB', 'PersonC', 'PersonD', 'PersonE']


def generate_CNN_model(train_data):
    X_train, Y_train = reformat_dataset(train_data, image_size=128)
    model = buildSequentialModel()

    if os.path.exists('../Trained Models/basic_CNN_Model.h5'):
        model = keras.models.load_model('../Trained Models/basic_CNN_Model.h5')
    else:
        model.fit(X_train, Y_train, validation_split=0.2, epochs=15)
        model.save('../Trained Models/basic_CNN_Model.h5')

    return model


def reformat_dataset(data, image_size):
    X = np.array([i[0] for i in data], dtype=object).reshape(-1, image_size, image_size, 1)
    Y = np.array([i[1] for i in data])
    Y = Y.reshape(len(Y), 5)

    X = np.asarray(X).astype(np.float32)
    return X, Y


def buildSequentialModel():
    model = tf.keras.Sequential([
        layers.Input(shape=(128, 128, 1)),
        layers.Conv2D(32, 5, strides=2, activation='relu'),
        layers.Conv2D(32, 5, activation='relu'),
        layers.MaxPool2D((3, 3)),
        layers.Conv2D(32, 5, strides=2, activation='relu'),
        layers.Conv2D(32, 5, activation='relu'),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(5, activation='softmax')
    ])

    model.summary()
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model


def test_model(test_data, model, visualise=False):
    X_test, Y_test = reformat_dataset(test_data, image_size=128)
    predictions = model.predict(X_test)
    y_true, predictions = reformat_labels(Y_test, predictions)
    print("Testing Accuracy: " + str(accuracy_score(y_true, predictions)))
    matrix = confusion_matrix(y_true, predictions)

    disp = ConfusionMatrixDisplay(confusion_matrix=matrix)
    disp.plot()
    plt.show()

    if visualise:
        Visualise_data(test_data, predictions)


def Visualise_data(data, predictions):
    for i in range(len(data)):
        plt.imshow(cv2.cvtColor(data[i][0], cv2.COLOR_BGR2RGB))
        plt.title("Prediction is " + str(image_Classes[predictions[i]]))
        plt.show()


def reformat_labels(y_true, y_test):
    true = []
    prediction = []
    for i in range(len(y_true)):
        true.append(np.argmax(y_true[i]))
        prediction.append(np.argmax(y_test[i]))
    return true, prediction
