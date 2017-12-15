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

# dataDir = 'Simple 4 letter dataset'
# dataName = '1000_4_letters_no_space_test.p'
dataDir = 'Nathan dataset'
dataName = '2000_4_letters_generalization.p'

# Load pickle data
data = pickle.load(open(os.path.join(dev_constants.MY_PROJECT_PATH, dataDir, dataName), 'rb'))
features = data[0]
labels = data[1]

width = 160
windowWidth = 70
slideDist = (width - windowWidth) / 3

# Load models
model1 = load_model('FINAL_four_letter_CNN_1_acc=97.h5')
model2 = load_model('FINAL_four_letter_CNN_2_acc=92.h5')
model3 = load_model('FINAL_four_letter_CNN_3_acc=88.h5')
model4 = load_model('FINAL_four_letter_CNN_4_acc=92.h5')

'''
models = [model1, model2, model3, model4]

string = []
for digit = range(4):
    windowStart = (int) ((digit-1) * slideDist)
    windowEnd = (int) (windowStart + windowWidth)
    X = preproc_image.cropWidth(features, windowStart, windowEnd)
    X = preproc_image.toGrayscale(X)
    letter = np.argmax(models[digit-1].predict(X), axis=1)
    string.append(letter)
    '''

# First letter
digit = 1
windowStart = (int) ((digit-1) * slideDist)
windowEnd = (int) (windowStart + windowWidth)
X = preproc_image.cropWidth(features, windowStart, windowEnd)
X = preproc_image.toGrayscale(X)
letter1 = np.argmax(model1.predict(X), axis=1)

# Second letter
digit = 2
windowStart = (int) ((digit-1) * slideDist)
windowEnd = (int) (windowStart + windowWidth)
X = preproc_image.cropWidth(features, windowStart, windowEnd)
X = preproc_image.toGrayscale(X)
letter2 = np.argmax(model2.predict(X), axis=1)

# Third letter
digit = 3
windowStart = (int) ((digit-1) * slideDist)
windowEnd = (int) (windowStart + windowWidth)
X = preproc_image.cropWidth(features, windowStart, windowEnd)
X = preproc_image.toGrayscale(X)
letter3 = np.argmax(model3.predict(X), axis=1)

# Fourth letter
digit = 4
windowStart = (int) ((digit-1) * slideDist)
windowEnd = (int) (windowStart + windowWidth)
X = preproc_image.cropWidth(features, windowStart, windowEnd)
X = preproc_image.toGrayscale(X)
letter4 = np.argmax(model4.predict(X), axis=1)

# Prediction
pred = np.transpose([letter1, letter2, letter3, letter4])
correctness = (np.sum(pred == labels, axis=1) == 4)

accuracy = sum(correctness) / len(correctness)
print('Accuracy is %f.' % accuracy)
