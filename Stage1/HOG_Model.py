import Data_Preparation
import HOG_utilities
import time

train_data, test_data = Data_Preparation.get_dataset(model_type='HOG', image_size=64)

training_start_time = time.time()
clf = HOG_utilities.generate_HOG_model(train_data)
training_stop_time = time.time()

testing_start_time = time.time()
HOG_utilities.testModel(test_data, clf, visualise=True)
testing_stop_time = time.time()

print(f"Training time: {training_stop_time - training_start_time}s")
print(f"Testing time: {testing_stop_time - testing_start_time}s")
