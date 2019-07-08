__author__ = 'ITI BANSAL'

'''
This file handles data creation, plotting and transformations
'''
import glob
import cv2
import h5py
import matplotlib.pyplot as plt
import numpy as np
import re
import transform as t
import tensorflow as tf
from keras.applications.vgg16 import VGG16
from sklearn.preprocessing import LabelEncoder
import sys

def create_dataset(path):
    '''
    This function creates hdf5 file from images
    :param path: directory containing input images
    '''
    addrs = glob.glob(path)
    hdf5_path = 'training_data.hdf5'
    train_shape = (len(addrs), 224, 224, 3)
    hdf5_file = h5py.File(hdf5_path, mode='w')
    labels = [re.split("_", addr)[2] for addr in addrs]
    hdf5_file.create_dataset("train_img", train_shape, np.int8)
    dt = h5py.special_dtype(vlen=str)
    hdf5_file.create_dataset("train_labels", (len(addrs),), dtype=dt)
    hdf5_file["train_labels"][...] = labels
    for i in range(len(addrs)):
        fname = addrs[i]
        img = cv2.imread(fname)
        img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_CUBIC)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        hdf5_file["train_img"][i, ...] = img[None]
    hdf5_file.close()

def plot_image(img):
    '''
    This creates plots of images
    '''
    #img = cv2.imread('/Users/itibansal/Downloads/handgesturedataset_part3/hand3_q_dif_seg_4_cropped.png')
    #print(img.shape)
    plt.imshow(img)
    plt.show()

def load_dataset(path):
    '''
    This method loads the dataset
    :param path: path of original hdf5 file
    :return: path of transformed hdf5 file
    '''
    hdf5_file = h5py.File(path, "r")
    train_x_orig = np.array(hdf5_file["train_img"][:])
    train_y_orig = np.array(hdf5_file["train_labels"][:])
    # train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    # Y_train = convert_to_one_hot(train_set_y_orig, 6).T
    labelencoder = LabelEncoder()
    train_y_orig = labelencoder.fit_transform(train_y_orig)
    train_y_orig = train_y_orig.reshape((1, train_y_orig.shape[0]))
    y_orig = convert_to_one_hot(train_y_orig, 36).T
    x_orig = train_x_orig / 255
    return x_orig,y_orig

def convert_to_one_hot(Y, C):
    Y = np.eye(C)[Y.reshape(-1)].T
    return Y

if __name__ == '__main__':
    path = "/Users/itibansal/Downloads/handgesturedataset_part3/*.png"
    create_dataset(path)
    path='training_data.hdf5'
    load_dataset(path)
    print('\n'.join(sys.path))

