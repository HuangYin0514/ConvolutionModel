# -*- coding: utf-8 -*-
# @Time     : 2019/1/17 22:18
# @Author   : HuangYin
# @FileName : MainWithResNets.py
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
from MethodWithResNets import *

if __name__ == '__main__':
    # tf.reset_default_graph()
    # with tf.Session() as session:
    #     np.random.seed(1)
    #     A_prev = tf.placeholder("float", [3, 4, 4, 6])
    #     X = np.random.randn(3, 4, 4, 6)
    #     A = identity_block(A_prev, f=2, filters=[2, 4, 6], stage=1, block='a')
    #     session.run(tf.global_variables_initializer())
    #     out = session.run([A], feed_dict={A_prev: X, K.learning_phase(): 0})
    #     print("out = " + str(out[0][1][1][0]))

    tf.reset_default_graph()
    with tf.Session() as session:
        np.random.seed(1)
        A_prev = tf.placeholder("float", [3, 4, 4, 6])
        X = np.random.randn(3, 4, 4, 6)
        A = convolutional_block(A_prev, f=2, filters=[2, 4, 6], stage=1, block='a')
        session.run(tf.global_variables_initializer())
        out = session.run([A], feed_dict={A_prev: X, K.learning_phase(): 0})
        print("out = " + str(out[0][1][1][0]))
