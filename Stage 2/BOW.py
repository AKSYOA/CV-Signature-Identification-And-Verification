from sklearn.metrics import accuracy_score
import numpy as np
import Data_Preparation
import BOW_utilities
from random import shuffle

train_images, test_images = Data_Preparation.get_dataset('BOW')


Classes = ['personA', 'personB', 'personC', 'personD', 'personE']

number_train_images = 40
number_test_images = 8
n_clusters = 300

for i in range(5):
    print(Classes[i], 'Signature Verification')

    train = train_images[i * number_train_images: (i + 1) * number_train_images]
    test = test_images[i * number_test_images: (i + 1) * number_test_images]

    model, k_means_object = BOW_utilities.generate_BOW_model(train, n_clusters)

    Y_test = BOW_utilities.get_labels(test)

    predictions = BOW_utilities.test_model(test, model, k_means_object, n_clusters)

    print("Testing Accuracy: " + str(accuracy_score(Y_test, predictions)))
