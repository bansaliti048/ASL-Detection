__author__ = 'ITI BANSAL'

'''
This file handles data creation, plotting and transformations
'''
import glob
import h5py
import numpy as np
import matplotlib.pyplot as plt
import cv2
import transform as t


def create_dataset(path):
    '''
    This function creates hdf5 file from images
    :param path: directory containing input images
    '''
    addrs = glob.glob(path)
    hdf5_path = 'training_data.hdf5'
    train_shape = (len(addrs), 224, 224, 3)
    hdf5_file = h5py.File(hdf5_path, mode='w')
    print(addrs)
    hdf5_file.create_dataset("train_img", train_shape, np.int8)
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

def transform_img(path):
    '''
    This function transforms the original image files
    :param path: path of original hdf5 file
    :return: path of transformed hdf5 file
    '''
    hdf5_file = h5py.File(path, "r")


if __name__ == '__main__':
    path = "/Users/itibansal/Downloads/handgesturedataset_part3/*.png"
    #create_dataset(path)
    path='training_data.hdf5'
    hdf5_file = h5py.File(path, "r")
    img = hdf5_file["train_img"][105, :]
    #image = Image.open("/Users/itibansal/Downloads/handgesturedataset_part3/hand3_q_dif_seg_4_cropped.png")
    plot_image(img)
    plot_image(t.hflip(img))
