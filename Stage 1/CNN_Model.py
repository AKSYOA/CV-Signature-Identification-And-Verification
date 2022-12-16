import Data_Preparation

X_train, Y_train, X_test, Y_test = Data_Preparation.get_dataset(image_size = 50)
print(X_train.shape)
print(Y_train.shape)