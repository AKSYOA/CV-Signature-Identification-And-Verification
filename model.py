import Data_Preparation
import cv2 as cv2
import matplotlib.pyplot as plt
import numpy as np
from sift import getSift


X_train, Y_train, X_test, Y_test = Data_Preparation.get_dataset()
