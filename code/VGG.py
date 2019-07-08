import tensorflow as tf
from keras.preprocessing import image
from keras.applications.vgg16 import VGG16
from matplotlib.pyplot import imshow
from keras.applications.imagenet_utils import preprocess_input
from keras.applications.vgg16 import decode_predictions
from keras.layers import Flatten,Dropout,Dense
from keras.layers import Input, Add, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, AveragePooling2D, MaxPooling2D, GlobalMaxPooling2D
from keras.models import Model
import numpy as np
import h5py
from sklearn.preprocessing import LabelEncoder
from keras.layers import Input, Dense
from keras.models import Model
import sys
import cv2
import os
import glob
import re

def load_dataset():
    #train_dataset = h5py.File('/Users/itibansal/Downloads/train_signs.h5', "r")
    train_dataset = h5py.File('/home/stu5/s16/ib6355/IndStudy/train_signs.h5', "r")
    train_set_x_orig = np.array(train_dataset["train_set_x"][:])  # your train set features
    train_set_y_orig = np.array(train_dataset["train_set_y"][:])  # your train set labels
    #test_dataset = h5py.File('/Users/itibansal/Downloads/test_signs.h5', "r")
    test_dataset = h5py.File('/home/stu5/s16/ib6355/IndStudy/test_signs.h5', "r")
    test_set_x_orig = np.array(test_dataset["test_set_x"][:])  # your test set features
    test_set_y_orig = np.array(test_dataset["test_set_y"][:])  # your test set labels
    train_x_orig = np.empty((len(train_set_x_orig), 224, 224, 3))
    test_x_orig = np.empty((len(test_set_x_orig), 224, 224, 3))
    for i in range(len(train_set_x_orig)):
        train_x_orig[i, :] = cv2.resize(train_set_x_orig[i, :], (224, 224), interpolation=cv2.INTER_CUBIC)
        # print(train_x_orig[i,:])
    for i in range(len(test_set_x_orig)):
        test_x_orig[i, :] = cv2.resize(test_set_x_orig[i, :], (224, 224), interpolation=cv2.INTER_CUBIC)
        # print(test_x_orig[i,:])
    classes = np.array(test_dataset["list_classes"][:])  # the list of classes
    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))
    return train_x_orig, train_set_y_orig, test_x_orig, test_set_y_orig, classes

def VGGModel():
    #Defining Model architechture
    model = VGG16(include_top=False, input_shape=(224, 224, 3))
    layer_dict = dict([(layer.name, layer) for layer in model.layers])
    x = layer_dict['block5_pool'].output
    x = Flatten()(x)
    x = Dense(4096, activation='relu')(x)
    x = Dense(4096, activation='relu')(x)
    x = Dense(6, activation='softmax')(x)
    #for layer in model.layers:
        #layer.trainable = False
    model = Model(input=model.input, output=x)
    # summarize
    model.summary()

    #Compiling model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def convert_to_one_hot(Y, C):
    Y = np.eye(C)[Y.reshape(-1)].T
    return Y

if __name__ == '__main__':
    X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes = load_dataset()
    # Normalize image vectors
    X_train = X_train_orig/255
    X_test = X_test_orig/255
    # Convert training and test labels to one hot matrices
    Y_train = convert_to_one_hot(Y_train_orig, 6).T
    Y_test = convert_to_one_hot(Y_test_orig, 6).T
    model=VGGModel()
    model.fit(X_train, Y_train, epochs=20, batch_size=32)
    preds = model.evaluate(X_test, Y_test)
    print("Loss = " + str(preds[0]))
    print("Test Accuracy = " + str(preds[1]))
    #model.save_weights("modelVGG1.h5")