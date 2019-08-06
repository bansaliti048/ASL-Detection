__author__ = 'ITI BANSAL'
'''
This program detects the American Sign Language from images using Residual Neural Networks (Convolution Neural Networks) 
The file accepts training data and test data as command line parameters in hdf5 format 
It gives several options of image pre-processing which can be selected by the user on execution
Usage: python3 ResNet.py train_data.hdf5 test_data.hdf5
'''

import numpy as np
from keras import layers
from keras.layers import Input, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, AveragePooling2D, MaxPooling2D
from keras.models import Model
from keras.initializers import glorot_uniform
from sklearn.preprocessing import LabelEncoder
from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K
import sys
import h5py


def identity_block(X, f, filters, stage, block):
    """
    Implementation of the identity block

    Arguments:
    X -- input tensor of shape (m, n_H_prev, n_W_prev, n_C_prev)
    f -- integer, specifying the shape of the middle CONV's window for the main path
    filters -- python list of integers, defining the number of filters in the CONV layers of the main path
    stage -- integer, used to name the layers, depending on their position in the network
    block -- string/character, used to name the layers, depending on their position in the network

    Returns:
    X -- output of the identity block, tensor of shape (n_H, n_W, n_C)
    """

    # defining name basis
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    # Retrieve Filters
    F1, F2, F3 = filters

    # Save the input value. You'll need this later to add back to the main path.
    X_shortcut = X

    # First component of main path
    X = Conv2D(filters=F1, kernel_size=(1, 1), strides=(1, 1), padding='valid', name=conv_name_base + '2a',
               kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '2a')(X)
    X = Activation('relu')(X)

    # Second component of main path (≈3 lines)
    X = Conv2D(filters=F2, kernel_size=(f, f), strides=(1, 1), padding='same', name=conv_name_base + '2b',
               kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '2b')(X)
    X = Activation('relu')(X)

    # Third component of main path (≈2 lines)
    X = Conv2D(filters=F3, kernel_size=(1, 1), strides=(1, 1), padding='valid', name=conv_name_base + '2c',
               kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '2c')(X)

    # Final step: Add shortcut value to main path, and pass it through a RELU activation (≈2 lines)
    X = layers.add([X, X_shortcut])
    X = Activation('relu')(X)

    return X


def convolutional_block(X, f, filters, stage, block, s=2):
    """
    Implementation of the convolutional block

    Arguments:
    X -- input tensor of shape (m, n_H_prev, n_W_prev, n_C_prev)
    f -- integer, specifying the shape of the middle CONV's window for the main path
    filters -- python list of integers, defining the number of filters in the CONV layers of the main path
    stage -- integer, used to name the layers, depending on their position in the network
    block -- string/character, used to name the layers, depending on their position in the network
    s -- Integer, specifying the stride to be used

    Returns:
    X -- output of the convolutional block, tensor of shape (n_H, n_W, n_C)
    """

    # defining name basis
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    # Retrieve Filters
    F1, F2, F3 = filters

    # Save the input value
    X_shortcut = X

    ##### MAIN PATH #####
    # First component of main path
    X = Conv2D(F1, (1, 1), strides=(s, s), padding='valid', name=conv_name_base + '2a',
               kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '2a')(X)
    X = Activation('relu')(X)

    # Second component of main path (≈3 lines)
    X = Conv2D(F2, (f, f), strides=(1, 1), padding='same', name=conv_name_base + '2b',
               kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '2b')(X)
    X = Activation('relu')(X)

    # Third component of main path (≈2 lines)
    X = Conv2D(F3, (1, 1), strides=(1, 1), padding='valid', name=conv_name_base + '2c',
               kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '2c')(X)

    ##### SHORTCUT PATH #### (≈2 lines)
    X_shortcut = Conv2D(F3, (1, 1), strides=(s, s), padding='valid', name=conv_name_base + '1',
                        kernel_initializer=glorot_uniform(seed=0))(X_shortcut)
    X_shortcut = BatchNormalization(axis=3, name=bn_name_base + '1')(X_shortcut)

    # Final step: Add shortcut value to main path, and pass it through a RELU activation (≈2 lines)
    X = layers.add([X, X_shortcut])
    X = Activation('relu')(X)

    return X


def ResNet50(input_shape=(64, 64, 3), classes=6):
    """
    Implementation of the popular ResNet50 the following architecture:
    CONV2D -> BATCHNORM -> RELU -> MAXPOOL -> CONVBLOCK -> IDBLOCK*2 -> CONVBLOCK -> IDBLOCK*3
    -> CONVBLOCK -> IDBLOCK*5 -> CONVBLOCK -> IDBLOCK*2 -> AVGPOOL -> TOPLAYER

    Arguments:
    input_shape -- shape of the images of the dataset
    classes -- integer, number of classes

    Returns:
    model -- a Model() instance in Keras
    """

    # Define the input as a tensor with shape input_shape
    X_input = Input(input_shape)

    # Zero-Padding
    X = ZeroPadding2D((3, 3))(X_input)

    # Stage 1
    X = Conv2D(64, (7, 7), strides=(2, 2), name='conv1', kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name='bn_conv1')(X)
    X = Activation('relu')(X)
    X = MaxPooling2D((3, 3), strides=(2, 2))(X)

    # Stage 2
    X = convolutional_block(X, f=3, filters=[64, 64, 256], stage=2, block='a', s=1)
    X = identity_block(X, 3, [64, 64, 256], stage=2, block='b')
    X = identity_block(X, 3, [64, 64, 256], stage=2, block='c')

    ### START CODE HERE ###

    # Stage 3 (≈4 lines)
    X = convolutional_block(X, f=3, filters=[128, 128, 512], stage=3, block='a', s=2)
    X = identity_block(X, 3, [128, 128, 512], stage=3, block='b')
    X = identity_block(X, 3, [128, 128, 512], stage=3, block='c')
    X = identity_block(X, 3, [128, 128, 512], stage=3, block='d')

    # Stage 4 (≈6 lines)
    X = convolutional_block(X, f=3, filters=[256, 256, 1024], stage=4, block='a', s=2)
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='b')
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='c')
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='d')
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='e')
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='f')

    # Stage 5 (≈3 lines)
    X = X = convolutional_block(X, f=3, filters=[512, 512, 2048], stage=5, block='a', s=2)
    X = identity_block(X, 3, [512, 512, 2048], stage=5, block='b')
    X = identity_block(X, 3, [512, 512, 2048], stage=5, block='c')

    # AVGPOOL (≈1 line). Use "X = AveragePooling2D(...)(X)"
    X = AveragePooling2D(pool_size=(2, 2), padding='same')(X)

    # output layer
    X = Flatten()(X)
    X = Dense(classes, activation='softmax', name='fc' + str(classes), kernel_initializer=glorot_uniform(seed=0))(X)

    # Create model
    model = Model(inputs=X_input, outputs=X, name='ResNet50')

    return model

def load_dataset(train_path,test_path):
    '''
    This method loads the dataset
    :param path: path of original hdf5 file
    :return: path of transformed hdf5 file
    '''
    hdf5_file = h5py.File(train_path, "r")
    train_x_orig = np.array(hdf5_file["train_img"][:])
    train_y_orig = np.array(hdf5_file["train_labels"][:])

    labelencoder = LabelEncoder()
    train_y_orig = labelencoder.fit_transform(train_y_orig)
    train_y_orig = train_y_orig.reshape((1, train_y_orig.shape[0]))
    y_train = convert_to_one_hot(train_y_orig, 36).T
    x_train = train_x_orig / 255
    hdf5_file.close()
    hdf5_file_test = h5py.File(test_path, "r")
    test_x_orig = np.array(hdf5_file_test["test_img"][:])
    test_y_orig = np.array(hdf5_file_test["test_labels"][:])

    labelencoder = LabelEncoder()
    test_y_orig = labelencoder.fit_transform(test_y_orig)
    test_y_orig = test_y_orig.reshape((1, test_y_orig.shape[0]))
    y_test = convert_to_one_hot(test_y_orig, 36).T
    x_test = test_x_orig / 255
    hdf5_file_test.close()
    return x_train,y_train,x_test,y_test

def convert_to_one_hot(Y, C):
    '''
    This function does the one hot encoding of the y labels
    '''
    Y = np.eye(C)[Y.reshape(-1)].T
    return Y


def predict(train_data,test_data):
    '''
    This function predicts the labels when the images are not centered
    '''
    # Load the data
    x_train, y_train, x_test, y_test = load_dataset(train_data, test_data)

    # Train the model and save weights
    model = ResNet50(input_shape=(224, 224, 3), classes=36)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(x_train, y_train, epochs = 20, batch_size = 32)
    model.save_weights("ResNetw1.h5")

    # Evaluate the model
    K.clear_session()
    x_train, y_train, x_test, y_test = load_dataset(train_data, test_data)

    # Train the model and save weights
    K.set_learning_phase(1)
    model = ResNet50(input_shape=(224, 224, 3), classes=36)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    # model.fit(x_train, y_train, epochs = 10, batch_size = 32)
    model.load_weights("ResNetw1.h5")
    preds = model.evaluate(x_test, y_test)
    print ("Loss = " + str(preds[0]))
    print ("Test Accuracy = " + str(preds[1]))


def predict_mean_image(train_data,test_data):
    '''
    This function predicts the labels when images are centered
    '''
    # Load the data
    x_train, y_train, x_test, y_test = load_dataset(train_data, test_data)
    datagen = ImageDataGenerator(featurewise_center=True)

    # Calculate mean on training dataset
    datagen.fit(x_train)

    # Prepare an iterators to scale images
    train_iterator = datagen.flow(x_train, y_train, batch_size=64)
    #print('Batches train=%d, test=%d' % (len(train_iterator), len(test_iterator)))

    # Train the nmodel
    model = ResNet50(input_shape=(224, 224, 3), classes=36)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit_generator(train_iterator, steps_per_epoch=len(train_iterator), epochs=20)
    model.save_weights("ResNetw2.h5")

    # Evaluate model
    K.clear_session()
    x_train, y_train, x_test, y_test = load_dataset(train_data, test_data)
    K.set_learning_phase(1)
    test_iterator = datagen.flow(x_test, y_test, batch_size=64)
    model = ResNet50(input_shape=(224, 224, 3), classes=36)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.load_weights("ResNetw2.h5")
    _, acc = model.evaluate_generator(test_iterator, steps=len(test_iterator), verbose=0)
    print('Test Accuracy: %.3f' % (acc * 100))

def predict_normalize_image(train_data,test_data):
    '''
    This function predicts the labels with normalized images
    '''
    x_train, y_train, x_test, y_test = load_dataset(train_data, test_data)

    model = ResNet50(input_shape=(224, 224, 3), classes=36)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Normalization
    datagen = ImageDataGenerator(rescale=1.0 / 255.0)
    train_iterator = datagen.flow(x_train, y_train, batch_size=64)
    test_iterator = datagen.flow(x_test, y_test, batch_size=64)
    print('Batches train=%d, test=%d' % (len(train_iterator), len(test_iterator)))
    batchX, batchy = train_iterator.next()
    print('Batch shape=%s, min=%.3f, max=%.3f' % (batchX.shape, batchX.min(), batchX.max()))

    # Train and save the model
    model.fit_generator(train_iterator, steps_per_epoch=len(train_iterator), epochs=20)
    model.save_weights("ResNetw3.h5")

    # Evaluate model
    K.clear_session()
    x_train, y_train, x_test, y_test = load_dataset(train_data, test_data)
    K.set_learning_phase(1)
    test_iterator = datagen.flow(x_test, y_test, batch_size=64)
    model = ResNet50(input_shape=(224, 224, 3), classes=36)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.load_weights("ResNetw3.h5")
    _, acc = model.evaluate_generator(test_iterator, steps=len(test_iterator), verbose=0)
    print('Test Accuracy: %.3f' % (acc * 100))

if __name__ == '__main__':
    try:
        train_data=sys.argv[1]
        test_data=sys.argv[2]
        print("Please enter the image processing technique to use:"
              "1) 1 for no preprocessing"
              "2) 2 for centering the images"
              "3) 3 for rescaling the images from [0-255] to [0-1]"
        option = input("Please enter the option: ")
        if option=="1":
            predict(train_data,test_data)
        elif option=="2":
            predict_mean_image(train_data,test_data)
        elif option=="3":
            predict_normalize_image(train_data,test_data)
        else:
            print("Enter options 1/2/3")
    except:
        print("Usage: python3 ResNet.py train_data.hdf5 test_data.hdf5")
