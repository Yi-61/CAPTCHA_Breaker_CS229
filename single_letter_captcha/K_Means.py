import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

import load_pickle_database

[dataset_read,label_read] = load_pickle_database.load_images_labels("/Users/yiliu/Box Sync/Stanford/Classes/2017-2018 Fall/CS 229 Machine Learning/Project/CAPTCHA Database/Simple single letter dataset/50000_single_letter.p")
print(dataset_read.shape)
print(label_read.shape)

nTrainPrcntg = 1
nTrain = int(nTrainPrcntg*dataset_read.shape[0])
trainData0 = dataset_read[0:nTrain,:]
trainLabel = label_read[0:nTrain,]
testData0 = dataset_read[nTrain:,:]
testLabel = label_read[nTrain:,]

#Perform PCA
PCAcontrol = 1
if PCAcontrol == 1:
    pca = PCA()
#     pca = PCA(n_components = 300)
    trainData = pca.fit_transform(trainData0)
else:
    trainData = trainData0
    testData = testData0
 
#K-means clustering
kmeans = KMeans(n_clusters = 26, verbose = 1)
kmeans.fit(trainData)
trainPredict = kmeans.predict(trainData)

#sort according to clusters (labels = 0,...,25)
order = np.argsort(trainPredict)
errors = -1*np.ones(26)
realLabels = -1*np.ones(26)
for labels in range(0,26):
    index = np.where(trainPredict == labels) #find all indexes with this label
    nCommonLabel = np.bincount(trainLabel[index]) #number of most common train label
    commonLabel = np.argmax(nCommonLabel) #find out the most common train label
    errors[labels] = np.asarray(index).shape[1] - nCommonLabel[commonLabel]
    realLabels[labels] = commonLabel 
print(errors)
print(realLabels)
accuracy = trainData.shape[0] - np.sum(errors)
print(accuracy)

#Output image and prediction
trainSelectedData =  trainData0[0:10,]
trainSelectedLabel =  trainPredict[0:10,]
for x in range(0,10):
    showData = trainSelectedData[x,].reshape(60,40)
    plt.imshow(showData, interpolation='nearest')
    labelName = 'Cluster label:' + str(realLabels[trainSelectedLabel[x,]])
    plt.title(labelName)
    plt.show()


# #plot
# dim1 = 0
# dim2 = 1 #dimensions of centroids that we will plot
# h = .02 #for mesh
# x_min, x_max = trainData[:, dim1].min() - 1, trainData[:, dim1].max() + 1
# y_min, y_max = trainData[:, dim2].min() - 1, trainData[:, dim2].max() + 1
# xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
# 
# #Obtain label for each point in mesh, using trained model
# Z = kmeans.predict(np.c_[xx.ravel(),yy.ravel()])
# 
# #plot the result into a color plot
# plt.figure(1)
# plt.clf()
# plt.imshow(Z, interpolation='nearest',extent=(xx.min(), xx.max(), yy.min(), yy.max()),
#            aspect='auto', origin='lower')
# plt.plot(trainData[0:dim1],trainData[:,dim2],'k.',markersize=2)
# 
# centroids = kmeans.cluster_centers_
# plt.scatter(centroids[:,dim1],centroids[:,dim2],marker = 'x',s=169, linewidths=3,color='w', zorder=10)
# plt.xlim(x_min, x_max)
# plt.ylim(y_min, y_max)
# plt.xticks(())
# plt.yticks(())
# plt.show()

