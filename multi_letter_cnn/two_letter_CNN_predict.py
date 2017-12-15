import pickle
import os
from keras.models import Sequential
from keras.layers import Dense,Dropout, Activation,Flatten
from keras.layers import Conv2D, MaxPooling2D
import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import load_model

import dev_constants
import preproc_image

dataDir = 'Simple 2 letter dataset'
dataName = '10000_2_letters_no_space_test.p'

# Load pickle data
data = pickle.load(open(os.path.join(dev_constants.MY_PROJECT_PATH, dataDir, dataName), 'rb'))
features = data[0]
labels = data[1]

# Convert to grayscale
X = preproc_image.toGrayscale(features)

# Load models
model1 = load_model('two_letter_CNN_1_acc=96.h5')
model2 = load_model('two_letter_CNN_2_acc=92.h5')

# Make predictions
letter1 = np.argmax(model1.predict(X), axis=1)
letter2 = np.argmax(model2.predict(X), axis=1)
pred = np.transpose([letter1, letter2])
correctness = (np.sum(pred == labels, axis=1) == 2)

accuracy = sum(correctness) / len(correctness)
print('Accuracy is %f.' % accuracy)
