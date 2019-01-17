# -*- coding: utf-8 -*-
# @Time     : 2019/1/17 14:33
# @Author   : HuangYin
# @FileName : MethodWithHappyHouse.py
# @Software : PyCharm

import numpy as np
from keras import layers
from keras.layers import Input, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D
from keras.layers import AveragePooling2D, MaxPooling2D, Dropout, GlobalAveragePooling2D, GlobalMaxPool2D
from keras.models import Model
from keras.preprocessing import image
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras.applications.imagenet_utils import preprocess_input
import pydot
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
from keras.utils import plot_model
from kt_utils import *

import keras.backend as K

K.set_image_data_format("channels_last")
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow


def model(input_shape):
    X_input = Input(input_shape)
    X = ZeroPadding2D((3, 3))(X_input)
    # conv
    X = Conv2D(32, (7, 7), strides=(1, 1), name="conv0")(X)
    X = BatchNormalization(axis=3, name="bn0")(X)
    X = Activation('relu')(X)
    # max pool
    X = MaxPooling2D((2, 2), name="max_pool")(X)
    # fc
    X = Flatten()(X)
    X = Dense(1, activation='sigmoid', name='fc')(X)
    # create model
    model = Model(inputs=X_input, outputs=X, name="HappyModel")

    return model


def HappyModel(input_shape):
    """
       Implementation of the HappyModel.

       Arguments:
       input_shape -- shape of the images of the dataset

       Returns:
       model -- a Model() instance in Keras
       """
    X_input = Input(input_shape)
    X = ZeroPadding2D((3, 3))(X_input)
    # conv
    X = Conv2D(32, (7, 7), strides=(1, 1), name="conv0")(X)
    X = BatchNormalization(axis=3, name="bn0")(X)
    X = Activation('relu')(X)
    # max pool
    X = MaxPooling2D((2, 2), name="max_pool")(X)
    # fc
    X = Flatten()(X)
    X = Dense(1, activation='sigmoid', name='fc')(X)
    # create model
    model = Model(inputs=X_input, outputs=X, name="HappyModel")

    return model
