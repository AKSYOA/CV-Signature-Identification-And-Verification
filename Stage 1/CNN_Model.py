from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import Data_Preparation
import tensorflow as tf
import numpy as np
from tensorflow.keras import layers
import os
import keras

X_train, Y_train, X_test, Y_test = Data_Preparation.get_dataset(model_type='CNN', image_size=128)
X_train = np.asarray(X_train).astype(np.float32)
X_test = np.asarray(X_test).astype(np.float32)


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


def reformat_labels(y_true, y_test):
    true = []
    prediction = []
    for i in range(len(y_true)):
        true.append(np.argmax(y_true[i]))
        prediction.append(np.argmax(y_test[i]))
    return true, prediction


def test_model():
    predictions = model.predict(X_test)
    y_true, predictions = reformat_labels(Y_test, predictions)
    print("Testing Accuracy: " + str(accuracy_score(y_true, predictions)))
    matrix = confusion_matrix(y_true, predictions)

    disp = ConfusionMatrixDisplay(confusion_matrix=matrix)
    disp.plot()
    plt.show()


model = buildSequentialModel()

if os.path.exists('../Trained Models/basicModel.h5'):
    model = keras.models.load_model('../Trained Models/basicModel.h5')
else:
    model.fit(X_train, Y_train, validation_split=0.2, epochs=15)
    model.save('../Trained Models/basicModel.h5')

test_model()
