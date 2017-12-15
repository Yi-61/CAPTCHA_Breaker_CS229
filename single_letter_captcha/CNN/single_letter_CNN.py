import pickle
import os
from keras.models import Sequential
from keras.layers import Dense,Dropout, Activation,Flatten
from keras.layers import Conv2D, MaxPooling2D
import numpy as np
from keras.utils import np_utils
# from PIL import Image
from sklearn.model_selection import train_test_split

import dev_constants
import preproc_image
import construct_CNN
import plot_acc_and_loss

numClass = 26
dropout = 0.2

dataDir = 'Simple single letter dataset'
dataName = '50000_single_letter.p'
outName = '50000_single_letter_CNN.h5'

# Load pickle data
data = pickle.load(open(os.path.join(dev_constants.MY_PROJECT_PATH, dataDir, dataName), 'rb'))
features = data[0]
labels = data[1]

# Convert labels to one-hot encoding
y = np_utils.to_categorical(labels,  num_classes = numClass)

# Convert to grayscale
X = preproc_image.toGrayscale(features)
inputShape = X[0].shape

# Train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

# Create the convolutional neural net
model = construct_CNN.constructModel_3(dropout, inputShape, numClass)

# Fit the model to the train
# careful about batch size, can lead to nonetype is not callable error
history = model.fit(X_train, y_train, validation_split = 0.2, batch_size = 400, epochs = 50, verbose = 1)

print(model.evaluate(X_test, y_test))
model.save(outName)

# Make plots
plot_acc_and_loss.draw(history)
