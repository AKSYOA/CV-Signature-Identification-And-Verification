from Stage2.Siamese_utilities import *
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from Stage2 import Data_Preparation


train_images, test_images = Data_Preparation.get_dataset()
train_triplets = get_triplet(train_images)
test_triplets = get_triplet(test_images)

siamese_model = generate_Siamese_model()

encoder = get_encoder((128, 128, 3))
encoder.load_weights('../Trained Models/encoder.h5')


def classify_images(signature_list1, signature_list2, threshold=1.3):
    tensor1 = encoder.predict(signature_list1)
    tensor2 = encoder.predict(signature_list2)

    distance = np.sum(np.square(tensor1 - tensor2), axis=-1)
    prediction = np.where(distance <= threshold, 0, 1)
    return prediction


def printAccuracy(positive_list, negative_list):
    true = np.array([0] * len(positive_list) + [1] * len(negative_list))
    predictions = np.append(positive_list, negative_list)

    print(f"\nAccuracy of model: {accuracy_score(true, predictions)}\n")


pos_list = np.array([])
neg_list = np.array([])

for data in get_batch(test_triplets, batch_size=30):
    a, p, n = data
    pos_list = np.append(pos_list, classify_images(a, p))
    neg_list = np.append(neg_list, classify_images(a, n))
    break

printAccuracy(pos_list, neg_list)
