import CNN_utilities
import Data_Preparation
import time


train_data, test_data = Data_Preparation.get_dataset(model_type='CNN', image_size=128)

training_start_time = time.time()
model = CNN_utilities.generate_CNN_model(train_data)
training_stop_time = time.time()

testing_start_time = time.time()
CNN_utilities.test_model(test_data, model, visualise=True)
testing_stop_time = time.time()


print(f"Training time: {training_stop_time - training_start_time}s")
print(f"Testing time: {testing_stop_time - testing_start_time}s")


