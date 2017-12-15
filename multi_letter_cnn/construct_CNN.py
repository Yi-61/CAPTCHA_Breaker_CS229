from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D

def constructModel_1(dropout, inputShape, numClass): # originally for single-letter
    model = Sequential()

    model.add(Conv2D(20, (2,2), strides = 1, activation = 'relu', input_shape = inputShape))
    model.add(Dropout(dropout))
    model.add(MaxPooling2D(pool_size = (2,2)))

    model.add(Conv2D(40, (2,2), strides = 1, activation = 'relu'))
    model.add(Dropout(dropout))
    model.add(MaxPooling2D(pool_size = (2,2)))

    model.add(Conv2D(80, (2,2), strides = 1, activation = 'relu'))
    model.add(Dropout(dropout))
    model.add(MaxPooling2D(pool_size = (2,2)))

    model.add(Conv2D(40, (2,2), strides = 1, activation = 'relu'))
    model.add(Dropout(dropout))
    model.add(MaxPooling2D(pool_size = (2,2)))

    model.add(Flatten())

    model.add(Dense(1000, activation = 'relu'))
    model.add(Dropout(dropout))
    model.add(Dense(100, activation = 'relu'))
    model.add(Dropout(dropout))

    model.add(Dense(numClass, activation = 'softmax'))

    model.compile(loss = 'categorical_crossentropy',
        optimizer = 'adam',
        metrics = ['accuracy'])
    return model

def constructModel_2(dropout, inputShape, numClass): # originally in multi-letter
    model = Sequential()

    model.add(Conv2D(20, (3,3), strides = 1, activation = 'relu', input_shape = inputShape))
    model.add(Dropout(dropout))
    model.add(MaxPooling2D(pool_size = (2,2)))

    model.add(Conv2D(40, (3,3), strides = 1, activation = 'relu'))
    model.add(Dropout(dropout))
    model.add(MaxPooling2D(pool_size = (2,2)))

    model.add(Conv2D(80, (3,3), strides = 1, activation = 'relu'))
    model.add(Dropout(dropout))
    model.add(MaxPooling2D(pool_size = (2,2)))

    model.add(Conv2D(40, (3,3), strides = 1, activation = 'relu'))
    model.add(Dropout(dropout))
    model.add(MaxPooling2D(pool_size = (2,2)))

    model.add(Flatten())

    model.add(Dense(1000, activation = 'relu'))
    model.add(Dropout(dropout))
    model.add(Dense(100, activation = 'relu'))
    model.add(Dropout(dropout))

    model.add(Dense(numClass, activation = 'softmax'))

    model.compile(loss = 'categorical_crossentropy',
        optimizer = 'adam',
        metrics = ['accuracy'])
    return model

def constructModel_3(dropout, inputShape, numClass): # Nathan used this, slightly modified
    model = Sequential()

    model.add(Conv2D(20, (2,2), strides = 1, activation = 'relu', input_shape = inputShape))
    model.add(MaxPooling2D(pool_size = (2,2)))

    model.add(Conv2D(40, (2,2), strides = 1, activation = 'relu'))
    model.add(Dropout(dropout))
    model.add(Conv2D(80, (2,2), strides = 1, activation = 'relu'))
    model.add(Dropout(dropout))
    model.add(Conv2D(40, (2,2), strides = 1, activation = 'relu'))
    model.add(Dropout(dropout))

    model.add(MaxPooling2D(pool_size = (2,2)))
    model.add(Flatten())

    model.add(Dense(500, activation = 'relu'))
    model.add(Dropout(dropout))
    model.add(Dense(200, activation = 'relu'))
    model.add(Dropout(dropout))

    model.add(Dense(numClass, activation = 'softmax'))

    model.compile(loss = 'categorical_crossentropy',
        optimizer = 'adam',
        metrics = ['accuracy'])
    return model

def changeDropout(model, dropout):
    x = model.layers[0].output
    for layer in model.layers[1:]:
        if 'Dropout' in str(type(layer)):
            x = Dropout(dropout)(x)
        else:
            x = layer(x)
    model = Model(inputs=model.inputs, outputs=x)

    model.compile(loss = 'categorical_crossentropy',
        optimizer = 'adam',
        metrics = ['accuracy'])
    return model
