import pickle
import os
import numpy as np
from keras.utils import np_utils
# from PIL import Image
from sklearn.model_selection import train_test_split
from keras.models import load_model

import dev_constants
import preproc_image
import construct_CNN
import plot_acc_and_loss

numClass = 26
dropout = 0.2
load_model_flag = False
change_dropout_flag = False

dataDir = 'Simple 2 letter dataset'
dataName = '10000_2_letters_no_space.p'
loadModelName = 'two_letter_CNN_2_acc=83.h5'

digit = 1
outName = 'two_letter_CNN_' + str(digit) + '.h5'

# Load pickle data
data = pickle.load(open(os.path.join(dev_constants.MY_PROJECT_PATH, dataDir, dataName), 'rb'))
features = data[0]
labels = data[1]
labels = labels[:,digit-1]

# Convert labels to one-hot encoding
y = np_utils.to_categorical(labels,  num_classes = numClass)

# Convert to grayscale
X = preproc_image.toGrayscale(features)
inputShape = X[0].shape

# train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

if load_model_flag:
    model = load_model(loadModelName)

    if change_dropout_flag: # change dropout values
        model = construct_CNN.changeDropout(model, dropout)
else:
    model = construct_CNN.constructModel_2(dropout, inputShape, numClass)

# Fit the model to the train
# Careful about batch size, can lead to nonetype is not callable error
history = model.fit(X_train, y_train, validation_split = 0.2, batch_size = 400, epochs = 50, verbose = 1)

print(model.evaluate(X_test, y_test))
model.save(outName);

# Make plots
plot_acc_and_loss.draw(history)
