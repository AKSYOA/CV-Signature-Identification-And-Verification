from sklearn.metrics import accuracy_score
import Data_Preparation
from Siamese_utilities import *
import matplotlib.pyplot as plt
import cv2

train_images, test_images = Data_Preparation.get_dataset()
train_triplets = get_triplet(train_images)
test_triplets = get_triplet(test_images)

siamese_network = get_siamese_network()
siamese_network.summary()