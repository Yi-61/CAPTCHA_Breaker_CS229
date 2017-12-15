from keras.models import load_model

model_name = 'single_char_VGG_transferred_CNN_massive_retrain.h5'

model = load_model(model_name);
print(model.summary())