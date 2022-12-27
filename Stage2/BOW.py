from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import Data_Preparation
import BOW_utilities
import os
import matplotlib.pyplot as plt
from joblib import dump, load

train_images, test_images = Data_Preparation.get_dataset()

Classes = ['personA', 'personB', 'personC', 'personD', 'personE']

number_train_images = 40
number_test_images = 8
n_clusters = 300

for i in range(5):
    print(Classes[i], 'Signature Verification')

    train = train_images[i * number_train_images: (i + 1) * number_train_images]
    test = test_images[i * number_test_images: (i + 1) * number_test_images]

    model_path = "../Trained Models/BOW_{person}_model.joblib".format(person=Classes[i])
    k_means_path = "../Trained Models/BOW_{person}_k_means.joblib".format(person=Classes[i])

    if os.path.exists(model_path):
        model = load(model_path)
        k_means_object = load(k_means_path)
    else:
        model, k_means_object = BOW_utilities.generate_BOW_model(train, n_clusters)

    Y_test = BOW_utilities.get_labels(test)

    predictions = BOW_utilities.test_model(test, model, k_means_object, n_clusters)

    print("Testing Accuracy: " + str(accuracy_score(Y_test, predictions)))

    matrix = confusion_matrix(Y_test, predictions)

    disp = ConfusionMatrixDisplay(confusion_matrix=matrix)
    disp.plot()
    plt.title(Classes[i])
    plt.show()
