from PIL import Image
import numpy as np

def toGrayscale(X):
    shape = list(np.shape(X))
    grayscaleX = np.empty(shape[:-1])
    for i in range(shape[0]):
        img = Image.fromarray(X[i].astype('uint8'), 'RGB')
        grayscaleImg = img.convert('L')
        grayscaleX[i] = np.array(grayscaleImg, dtype = 'uint8')
    shape[-1] = 1
    grayscaleX = grayscaleX.reshape(shape)
    return grayscaleX

def cropWidth(X, start, end):
    shape = list(np.shape(X))
    shape[2] = end - start
    croppedX = np.empty(shape)
    for i in range(shape[0]):
        img = Image.fromarray(X[i].astype('uint8'), 'RGB')
        croppedImg = img.crop((start, 0, end, 60))
        croppedX[i] = np.array(croppedImg, dtype = 'uint8')
    return croppedX
