import CNN_utilities
import Data_Preparation


train_data, test_data = Data_Preparation.get_dataset(model_type='CNN', image_size=128)

model = CNN_utilities.generate_CNN_model(train_data)

CNN_utilities.test_model(test_data, model)





