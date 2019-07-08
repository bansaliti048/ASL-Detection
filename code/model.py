__author__ = 'ITI BANSAL'

import tensorflow as tf
from keras.preprocessing import image
from keras.applications.vgg16 import VGG16
from matplotlib.pyplot import imshow
from keras.applications.imagenet_utils import preprocess_input
from keras.applications.resnet50 import ResNet50
from keras.applications.inception_v3 import InceptionV3
from keras.models import Model, load_model


img = image.load_img('/Users/itibansal/Downloads/dog.jpg', target_size=(224, 224))
imshow(img)
img = image.img_to_array(img)
img = preprocess_input(img)

#VGG
# model=VGG16(include_top=False, weights=None)
# # summarize the model
# model.summary()
# model.load_weights('/Users/itibansal/Downloads/vgg16_weights.h5')
# print(model.optimizer)
# features = model.predict(img)

#ResNet
model = ResNet50(weights='http://github.com/fchollet/deep-learning-models/releases/download/v0.2/resnet50_weights_tf_dim_ordering_tf_kernels.h5')