from sklearn.metrics import accuracy_score
import Data_Preparation
from Siamese_utilities import *
from tensorflow.keras.optimizers import Adam

import matplotlib.pyplot as plt
import cv2

train_images, test_images = Data_Preparation.get_dataset()
train_triplets = get_triplet(train_images)
test_triplets = get_triplet(test_images)

siamese_network = get_siamese_network()
siamese_network.summary()

siamese_model = SiameseModel(siamese_network)
optimizer = Adam(learning_rate=1e-3, epsilon=1e-01)
siamese_model.compile(optimizer=optimizer)
