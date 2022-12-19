from sklearn.metrics import accuracy_score
import Data_Preparation
from siamese_utilites import *
import matplotlib.pyplot as plt
import cv2
f, axes = plt.subplots(1, 3, figsize=(15, 20))
train_triplets, test_triplets = Data_Preparation.get_dataset('siames')
for i in get_batch(train_triplets, 256):
     a,b,c=i
     a = np.array(a[0], dtype=object)
     b = np.array(b[0], dtype=object)
     c = np.array(c[0], dtype=object)

