import Data_Preparation
from sklearn.metrics import confusion_matrix, accuracy_score, ConfusionMatrixDisplay
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt

X_train, Y_train, X_test, Y_test = Data_Preparation.get_dataset(model_type='HOG', image_size=64)

clf = LogisticRegression().fit(X_train, Y_train)

predictions = clf.predict(X_test)

print("Testing Accuracy: " + str(accuracy_score(Y_test, predictions)))

matrix = confusion_matrix(Y_test, predictions)

disp = ConfusionMatrixDisplay(confusion_matrix=matrix)
disp.plot()
plt.show()
