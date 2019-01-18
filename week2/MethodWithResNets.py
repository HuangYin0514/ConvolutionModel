# -*- coding: utf-8 -*-
# @Time     : 2019/1/17 22:26
# @Author   : HuangYin
# @FileName : MethodWithResNets.py
# @Software : PyCharm
import numpy as np
from keras import layers
from keras.layers import Input, Add, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, \
    AveragePooling2D, MaxPooling2D, GlobalMaxPooling2D
from keras.models import Model, load_model
from keras.preprocessing import image
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras.applications.imagenet_utils import preprocess_input
import pydot
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
from keras.utils import plot_model
from resnets_utils import *
from keras.initializers import glorot_uniform
import scipy.misc
from matplotlib.pyplot import imshow

import keras.backend as K

K.set_image_data_format('channels_last')
K.set_learning_phase(1)


def identity_block(X, f, filters, stage, block):
    """
       Implementation of the identity block as defined in Figure 3

       Arguments:
       X -- input tensor of shape (m, n_H_prev, n_W_prev, n_C_prev)
       f -- integer, specifying the shape of the middle CONV's window for the main path
       filters -- python list of integers, defining the number of filters in the CONV layers of the main path
       stage -- integer, used to name the layers, depending on their position in the network
       block -- string/character, used to name the layers, depending on their position in the network

       Returns:
       X -- output of the identity block, tensor of shape (n_H, n_W, n_C)
       """
    conv_name_base = "res" + str(stage) + block + "_branch"
    bn_name_base = "bn" + str(stage) + block + "_branch"

    F1, F2, F3 = filters
    X_shortcut = X

    # Frist component of main path
    X = Conv2D(filters=F1, kernel_size=(1, 1), strides=(1, 1), padding='valid', name=conv_name_base + '2a',
               kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '2a')(X)
    X = Activation('relu')(X)

    # second component of main path
    X = Conv2D(filters=F2, kernel_size=(f, f), strides=(1, 1), padding='same',
               name=conv_name_base + '2b',
               kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '2b')(X)
    X = Activation('relu')(X)

    # third component of main path
    X = Conv2D(filters=F3, kernel_size=(1, 1), strides=(1, 1), padding='valid',
               name=conv_name_base + '2c',
               kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '2c')(X)
    X = Activation('relu')(X)

    # final step : add shortcut value to main path , and pass it  though a Relu activation
    X = Add()([X, X_shortcut])
    X = Activation('relu')(X)

    return X


def convolutional_block(X, f, filters, stage, block, s=2):
    """
       Implementation of the convolutional block as defined in Figure 4

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

    conv_name_base = "res" + str(stage) + block + "_branch"
    bn_name_base = "bn" + str(stage) + block + "_branch"

    F1, F2, F3 = filters
    X_shortcut = X

    # Frist component of main path
    X = Conv2D(filters=F1, kernel_size=(1, 1), strides=(s, s), padding='valid', name=conv_name_base + '2a',
               kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '2a')(X)
    X = Activation('relu')(X)

    # second component of main path
    X = Conv2D(filters=F2, kernel_size=(f, f), strides=(1, 1), padding='same',
               name=conv_name_base + '2b',
               kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '2b')(X)
    X = Activation('relu')(X)

    # third component of main path
    X = Conv2D(filters=F3, kernel_size=(1, 1), strides=(1, 1), padding='valid',
               name=conv_name_base + '2c',
               kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '2c')(X)
    X = Activation('relu')(X)

    # match up same dim with final main path
    X_shortcut = Conv2D(filters=F3, kernel_size=(1, 1), strides=(s, s), padding='valid',
                        name=conv_name_base + '1',
                        kernel_initializer=glorot_uniform(seed=0))(X_shortcut)
    X_shortcut = BatchNormalization(axis=3, name=bn_name_base + '1')(X_shortcut)

    # final step : add shortcut value to main path , and pass it  though a Relu activation
    X = Add()([X, X_shortcut])
    X = Activation('relu')(X)

    return X
