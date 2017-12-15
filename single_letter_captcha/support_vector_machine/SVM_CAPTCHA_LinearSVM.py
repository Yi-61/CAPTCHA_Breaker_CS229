import numpy as np
from PIL import Image
from sklearn import svm
import os
import pickle

import load_pickle_database
from numpy import int

[dataset_read,label_read] = load_pickle_database.load_images_labels("/Users/yiliu/Box Sync/Stanford/Classes/2017-2018 Fall/CS 229 Machine Learning/Project/CAPTCHA Database/Simple single letter dataset/50000_single_letter.p")

#Split train set and test set
nTrainPrcntg = 0.7
nTrain = int(nTrainPrcntg*dataset_read.shape[0])
trainData = dataset_read[0:nTrain,:]
trainLabel = label_read[0:nTrain,]
testData = dataset_read[nTrain:,:]
testLabel = label_read[nTrain:,]

#Train model
multiClassSVM = svm.LinearSVC(loss = 'squared_hinge', penalty = 'l2',C=1/5000 ,max_iter=8000,verbose=1) #define kernel? default should be rbf-kernel
multiClassSVM.fit(trainData,trainLabel)
saveFile = "/Users/yiliu/Box Sync/Stanford/Classes/2017-2018 Fall/CS 229 Machine Learning/Project/CAPTCHA Database/Simple single letter dataset/SVM_Model.p"
pickle.dump(multiClassSVM,open(os.path.join(saveFile),'wb')) #Save model
print("Model Saved")
# print(trainData.shape)

#Load prediction dataset
loadPath = "/Users/yiliu/Box Sync/Stanford/Classes/2017-2018 Fall/CS 229 Machine Learning/Project/CAPTCHA Database/Simple single letter dataset/prediction_data.p"
[predData,predLabel] = pickle.load(open(loadPath, 'rb'))

#Training accuracy
confidenceTrain = multiClassSVM.decision_function(trainData)
predictionTrain = np.argmax(confidenceTrain, axis=1)
print(trainLabel.shape)
comparisonTrain = np.equal(predictionTrain,trainLabel)
accuracyTrain = sum(comparisonTrain)/trainLabel.size
print('Train accuracy:')
print(accuracyTrain)
 
#Test accuracy
confidenceTest = multiClassSVM.decision_function(testData)
predictionTest = np.argmax(confidenceTest, axis=1)
comparisonTest = np.equal(predictionTest,testLabel)
accuracyTest = sum(comparisonTest)/testLabel.size
print('Test accuracy:')
print(accuracyTest)

#Make predictions
confidencePred = multiClassSVM.decision_function(predData)
predictionPred = np.argmax(confidencePred, axis=1)

#Prediction accuracy
temp = 0
iteration = 0
for x in range(0,10):
    iteration = iteration + 1
    if predictionPred[x] == ord(predLabel[x,])-65:
        temp = temp+1
print('Correct number:')
print(temp)

accuracyPred = temp/predLabel.size
print('Prediction accuracy:')
print(accuracyPred)
