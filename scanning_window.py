# from PIL import Image
from io import BytesIO
from captcha.image import ImageCaptcha
from random import randint
import os
import numpy as np
import pickle
from keras.models import load_model

import dev_constants
import preproc_image

stringLen = 4
exampleNum = 1000
dataDir = dev_constants.MY_PROJECT_PATH + ('\\Simple ' + str(stringLen) + ' letter dataset\\')
loadModelName = 'single_letter_CNN_acc=98.h5'

if stringLen == 2: # can achieve 80.5% accuracy with 2-letter CAPTCHA
    width = 100
    threshold = 6
    bandWidth = 9
    halfWidth = (int) (bandWidth/2)
    filterWidth = 15
    gap = 25
else: # can achieve 38.0% accuracy with 4-letter CAPTCHA
    width = 160
    threshold = 7
    bandWidth = 9
    halfWidth = (int) (bandWidth/2)
    filterWidth = 15
    gap = 20

windowWidth = 40

separate = False
if separate:
    dataName = str(exampleNum) + '_' + str(stringLen) + '_letters_with_space_test.p'
else:
    dataName = str(exampleNum) + '_' + str(stringLen) + '_letters_no_space_test.p'

data = pickle.load(open(os.path.join(dev_constants.MY_PROJECT_PATH, dataDir, dataName), 'rb'))

features = data[0]
labels = data[1]
X = preproc_image.toGrayscale(features)
# X = features

model = load_model(loadModelName)

correctNum = 0
for exampleIndex in range(exampleNum):
    x = X[exampleIndex]
    label = labels[exampleIndex]

    # img = Image.fromarray(features[exampleIndex].astype('uint8'), 'RGB')
    # img.show()

    predictions = []
    for windowLeft in range(width - windowWidth):
        windowRight = windowLeft + windowWidth
        window = x[:, windowLeft:windowRight, :]
        predictions.append(model.predict(np.array([window]))[0])

    predictions = np.array(predictions)

    string = []
    prob = []
    filterStart = 0
    while filterStart + filterWidth < width - windowWidth:
        predSubset = predictions[filterStart : filterStart+filterWidth, :]

        maxProbIndex = np.argmax(predSubset)
        centerPos, centerLetter = np.unravel_index(maxProbIndex, predSubset.shape)
        maxProb = predSubset[centerPos, centerLetter]
        centerPos += filterStart
        # print(maxProbIndices)
        # print(predSubset[maxProbIndices[0], maxProbIndices[1]])

        minIndex = max(0, centerPos-halfWidth)
        maxIndex = min(width, centerPos+halfWidth+1)
        pred = np.argmax(predictions[minIndex:maxIndex, :], axis=1)
        if sum(pred == centerLetter) >= threshold:
            filterStart = centerPos + gap
            string.append(centerLetter)
            prob.append(maxProb)
            # print('%c at position %d.' % (chr(centerLetter+ord('A')), centerPos))
        else:
            filterStart += 1

    # print(''.join([chr(i+ord('A')) for i in string]))
    # print(''.join([chr(i+ord('A')) for i in label]))
    string = np.array(string)
    if len(string) > stringLen:
        prob = np.array(prob)
        filteredString = (prob.argsort() >= len(prob)-stringLen) * (string + 1) - 1
        string = list(filter(lambda a: a >=0 , filteredString))

    if len(string) == stringLen:
        correctNum += (sum(string == label) == stringLen)
    if (exampleIndex+1) % 10 == 0:
        print('%d completed, %d correct, accuracy = %f.' % (exampleIndex+1, correctNum, correctNum/(exampleIndex+1)))

# Calculate accuracy
accuracy = correctNum / exampleNum
print('Accuracy is %f.' % accuracy)
