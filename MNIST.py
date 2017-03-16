from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys

import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K

#number of samples that will be sent through the network
batch_size = 128

#I'm not too sure what classes are, but my guess is that they are the different "things" the classifier chooses to classify
#Nevermind, I think this is the digits 0-9. I'll leave the above comment up for reference
num_classes = 10
#epoch = one forward pass and one backward pass of all the training examples
#12 runthroughs of the training examples
epochs = 12

#matrix dimensions for the image that contains the MNIST dataset. The specific dimensions are abitrary
img_rows, img_cols = 28, 28
#x_train/test are the image data that are the initial inputs
#y_train/test represent the outputs
(x_train, y_train), (x_test, y_test) = mnist.load_data()

#I'm guessing this is checking if the processing is using image data from the backend using the channels as the first priority
if K.image_data_format() == 'channels_first':
    #sets x_train to a shape that it's supposed to be. It's training x to be 0
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    #tests if x recognized it is 0
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
#This checks if its not the channels. I think that means that the "1" is related to channels somehow
else:

    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)

    #This sets the shape we are training the model for as well as testing it to the image.
    input_shape = (img_rows, img_cols, 1)

#changes the x values to floats
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
#sets the colors back to their integer representations
x_train /= 255
x_test /= 255

#creates the matrix that will set the output to the classes
#[[y1, 0], [y2, 1], ... ,[y10, 9]]
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

#"Sequential is a linear stack of layers" - Keras documentation
model = Sequential()
#Just reading this, I'm assuming we are creating the first convolutional layer.
#32 may have something to do with the depth of the layer because the next layer grows by a factor of 2
#kernel size is the size of the matrix kernel
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape))
model.add(Conv2D(64, (3, 3), activation='relu'))
#2D represents the fact that we're dealing with pooling data
#"Max pooling is a sample-based discretization process.
# The objective is to down-sample an input representation (image, hidden-layer output matrix, etc.),
# reducing it's dimensionality and allowing for assumptions to be made about features contained in the sub-regions binned."
# - quora
#Basically, it breaks down the abstract (image) into pieces (in this case, 2x2 pixels?)
model.add(MaxPooling2D(pool_size=(2, 2)))
#prevents overfitting
model.add(Dropout(0.25))
#Based on what I read in the Keras documentation, I think this flattens the model to one dimension
model.add(Flatten())
#"A dense layer is a kind of hidden layer where every node is connected to every other node in the next layer." - Quora
#This adds a dense layer
#The 128 possibly represents all the connections in an image with 256 values for pixels
model.add(Dense(128, activation='relu'))
#Dropout for the first layer
model.add(Dropout(0.5))
#Densifies the layer to connect to the output of 10 digits
model.add(Dense(num_classes, activation='softmax'))

#Compiles
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

#Fits data to model?
model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs,
          verbose=1, validation_data=(x_test, y_test))
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
