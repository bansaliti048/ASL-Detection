__author__ = 'ITI BANSAL'

'''
This file contains methods for image transformation
'''

import numpy as np
from PIL import Image, ImageEnhance

def transform(type,img):
    if type=="hflip":
        return hflip(img)
    elif type=="enhance":
        return enhance_image(img)

def hflip(img):
    '''
    This function horizontally flips an image
    '''
    return np.fliplr(img)

def crop(img,xmin,xmax,ymin,ymax):
    #img = cv2.imread('/Users/itibansal/Downloads/handgesturedataset_part3/hand3_q_dif_seg_4_cropped.png')
    '''
    This function crops an image
    :param img: input image
    :param xmin: left pixel on x-axis
    :param xmax: right pixel on x-axis
    :param ymin: top pixel of an image
    :param ymax: bottom most pixel of an image
    '''
    img = img[xmin:xmax,ymin:ymax]
    return img

def enhance_image(img):
    #image = Image.open("/Users/itibansal/Downloads/handgesturedataset_part3/hand3_q_dif_seg_4_cropped.png")
    image = Image.fromarray(img.astype('uint8'))
    contrast = ImageEnhance.Color(image)
    # image.show()
    return contrast.enhance(2)