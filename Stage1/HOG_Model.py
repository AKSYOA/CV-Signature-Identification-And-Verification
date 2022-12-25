import Data_Preparation
import HOG_utilities

train_data, test_data = Data_Preparation.get_dataset(model_type='HOG', image_size=64)
clf = HOG_utilities.generate_HOG_model(train_data)

HOG_utilities.testModel(test_data, clf, visualise=True)

