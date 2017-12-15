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
dropout = 0.4
load_model_flag = True
change_dropout_flag = True

dataDir = 'Simple 4 letter dataset'
dataName = '50000_4_letters_no_space.p'
loadModelName = 'NEW_four_letter_CNN_3_acc=86.h5'

digit = 3
outName = 'four_letter_CNN_' + str(digit) + '.h5'

# Load pickle data
data = pickle.load(open(os.path.join(dev_constants.MY_PROJECT_PATH, dataDir, dataName), 'rb'))
features = data[0]
labels = data[1]
labels = labels[:,digit-1]

# Convert labels to one-hot encoding
y = np_utils.to_categorical(labels,  num_classes = numClass)

width = 160
windowWidth = 70
slideDist = (width - windowWidth) / 3
windowStart = (int) ((digit-1) * slideDist)
windowEnd = (int) (windowStart + windowWidth)

# Convert to grayscale
X = preproc_image.cropWidth(features, windowStart, windowEnd)
X = preproc_image.toGrayscale(X)
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
history = model.fit(X_train, y_train, validation_split = 0.2, batch_size = 400, epochs = 60, verbose = 1)

print(model.evaluate(X_test, y_test))
model.save(outName)

# Make plots
plot_acc_and_loss.draw(history)
