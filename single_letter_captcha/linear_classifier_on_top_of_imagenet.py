from keras import applications
from sklearn.linear_model import LogisticRegression
from keras import optimizers
from keras.models import Sequential, Model
from keras.layers import Input, Dropout, Flatten, Dense, GlobalAveragePooling2D
from keras.layers import Conv2D
from keras import backend as k
from keras.applications.inception_v3 import InceptionV3
import numpy as np
from sklearn.model_selection import train_test_split
import dev_constants as dev
from keras.utils import np_utils
import load_pickle_database as pdb

img_width, img_height = 256, 256
train_data_dir = "data/train"
validation_data_dir = "data/val"
## test the model on our captchas data-set
dir= dev.MY_PROJECT_PATH+'\\ConvolutionalNeuralNets\\'
dir= dev.MY_PROJECT_PATH+'\\TransferLearningWithKeras\\'

path = dir+'single_char_resized.p'

images, features = pdb.load_images_labels(path);
X = images;
# X = X[:,:,:,0:2];
X = np.reshape(X, (50000,139, 139,3));
labels = np.array([ord(i) for i in features]);
labels = labels - 65;
y = np_utils.to_categorical(labels);

# model = applications.VGG16(weights = "imagenet", include_top=False,\
#                            input_shape = (img_width, img_height, 3))
model = applications.Xception(weights = "imagenet", include_top=False,\
                           input_shape = (139,139, 3), pooling = 'max')


## add custom layers (which we tune so we can train on our captchas)
## one of the layers that we must change is the input layer!
input = Input(shape =  (139,139,3), name = 'captcha_input')
output_vgg16_conv = model(input);
## add in some extra layers into the output


# creating the final model
model_final = Model(input = input, output = output_vgg16_conv)

# compile the model
model_final.compile(loss = "categorical_crossentropy",\
                    optimizer = optimizers.SGD(lr=0.0001, momentum=0.9), metrics=["accuracy"])
print(model_final.summary())

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2);

X_train = model_final.predict(X_train, batch_size = 200, verbose=1);
X_test = model_final.predict(X_test, batch_size = 100, verbose=1);
print(X_train.shape)
resize = np.prod(X_train.shape[1:])

X_train = np.reshape(X_train, (40000, resize));
X_test = np.reshape(X_test, (10000, resize));
##further feature snipping
bf_train_s = X_train
bf_test_s = X_test

print(X_train.shape);
print(X_test.shape)

## train a linear classifier:
y_train_mc = np.argmax(y_train, axis=1)
y_test_mc = np.argmax(y_test, axis = 1)

clf = LogisticRegression(multi_class='multinomial', solver='lbfgs', verbose=1)
clf.fit(bf_train_s,y_train_mc);
print(clf.score(bf_test_s, y_test_mc));


