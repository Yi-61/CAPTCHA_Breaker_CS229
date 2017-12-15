'''
Created on Nov 13, 2017

@author: yiliu
'''

import numpy as np
from PIL import Image
from sklearn import svm
import os
import pickle

import load_pickle_database
from numpy import int

[dataset_read,label_read] = load_pickle_database.load_images_labels("/Users/yiliu/Box Sync/Stanford/Classes/2017-2018 Fall/CS 229 Machine Learning/Project/CAPTCHA Database/Simple single letter dataset/50000_single_letter.p")
# dataLabel = dataLabel.reshape(-1)
print(dataset_read.shape[0])
# print(np.nonzero(label_read))
# nTrain=70
nTrainPrcntg = 0.7
nTrain = int(nTrainPrcntg*dataset_read.shape[0])
trainData = dataset_read[0:nTrain,:]
trainLabel = label_read[0:nTrain,]
multiClassSVM = svm.LinearSVC(loss = 'squared_hinge', penalty = 'l2',C=1/5000 ,max_iter=8000,verbose=1) #define kernel? default should be rbf-kernel
# multiClassSVM = svm.LinearSVC(verbose=1,max_iter=5000)
multiClassSVM.fit(trainData,trainLabel)
saveFile = "/Users/yiliu/Box Sync/Stanford/Classes/2017-2018 Fall/CS 229 Machine Learning/Project/CAPTCHA Database/Simple single letter dataset/SVM_Model.p"
pickle.dump(multiClassSVM,open(os.path.join(saveFile),'wb'))
print("Model Saved")
print(trainData.shape)

testData = dataset_read[nTrain:,:]
testLabel = label_read[nTrain:,]

loadPath = "/Users/yiliu/Box Sync/Stanford/Classes/2017-2018 Fall/CS 229 Machine Learning/Project/CAPTCHA Database/Simple single letter dataset/prediction_data.p"
[predData,predLabel] = pickle.load(open(loadPath, 'rb'))
# predLabel = np.asarray(predLabel)

#Training accuracy
confidenceTrain = multiClassSVM.decision_function(trainData)
print(confidenceTrain.shape)
predictionTrain = np.argmax(confidenceTrain, axis=1)
print(predictionTrain.shape)
print(predictionTrain[[0,],])
print(trainLabel.shape)
comparisonTrain = np.equal(predictionTrain,trainLabel)
accuracyTrain = sum(comparisonTrain)/trainLabel.size
# print(comparisonTrain.shape)
print('Train accuracy:')
print(accuracyTrain)
 
#Test accuracy
confidenceTest = multiClassSVM.decision_function(testData)
print(confidenceTest.shape)
predictionTest = np.argmax(confidenceTest, axis=1)
print(predictionTest.shape)
print(predictionTest[[0,],])
comparisonTest = np.equal(predictionTest,testLabel)
accuracyTest = sum(comparisonTest)/testLabel.size
print('Test accuracy:')
print(accuracyTest)

#Pred accuracy
confidencePred = multiClassSVM.decision_function(predData)
print(confidencePred.shape)
predictionPred = np.argmax(confidencePred, axis=1)
print(predictionPred.shape)
print(predictionPred[0])
print(predLabel.shape)
# comparisonPred = np.equal(np.asarray(predictionPred),np.asarray(predLabel),out='ndarray')
print(type(predictionPred))
print(type(predLabel))
# comparisonPred = np.array(predictionPred) == np.array(predLabel)

temp = 0
iteration = 0
for x in range(0,10):
    iteration = iteration + 1
#     print(predictionPred[0])
    if predictionPred[x] == ord(predLabel[x,])-65:
#         print(predictionPred[x])
#         print(predLabel[x,])
        temp = temp+1
print('Correct number:')
print(temp)

# print('test output:')
# print(np.asarray(comparisonPred).shape)
# print('confirm')
# print(comparisonPred)
print(predLabel.size)
# accuracyPred = sum(comparisonPred)/predLabel.size
accuracyPred = temp/predLabel.size
print('Pred accuracy:')
print(accuracyPred)



#Train data 900, test data 100, train accuracy 100%, test accuracy 22.0%
#Train data 4500, test data 500, train accuracy 95.1%, test accuracy 21%
