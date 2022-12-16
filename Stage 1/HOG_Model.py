import Data_Preparation

X_train, Y_train, X_test, Y_test = Data_Preparation.get_dataset(model_type='HOG', image_size=64)

print(X_train.shape)
print(Y_train.shape)
