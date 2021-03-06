# -*- coding: utf-8 -*-
# @Time     : 2019/1/15 13:28
# @Author   : HuangYin
# @FileName : MainWithTensorFlow.py
# @Software : PyCharm

import math
import numpy as py
import h5py
import matplotlib.pyplot as plt
import scipy
from PIL import Image
from scipy import ndimage
import tensorflow as tf
from tensorflow.python.framework import ops
from cnn_utils import *
from ConvWithTensorFlowMethod import *

if __name__ == '__main__':
    # keep same random
    np.random.seed(1)

    # load data
    X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes = load_dataset()

    # Example of a picture
    index = 3
    plt.imshow(X_train_orig[index])
    print("y = " + str(np.squeeze(Y_train_orig[:, index])))
    # plt.show()

    # normalize
    X_train = X_train_orig / 255
    X_test = X_test_orig / 255
    Y_train = convert_to_one_hot(Y_train_orig, 6).T
    Y_test = convert_to_one_hot(Y_test_orig, 6).T
    print("number of training examples = " + str(X_train.shape[0]))
    print("number of test examples = " + str(X_test.shape[0]))
    print("X_train shape: " + str(X_train.shape))
    print("Y_train shape: " + str(Y_train.shape))
    print("X_test shape: " + str(X_test.shape))
    print("Y_test shape: " + str(Y_test.shape))
    conv_layers = {}

    # test create_placeholders
    X, Y = create_placeholders(64, 64, 3, 6)
    print("X = " + str(X))
    print("Y = " + str(Y))

    # test initialize_parameters
    tf.reset_default_graph()
    with tf.Session() as session:
        parameters = initialize_parameters()
        init = tf.global_variables_initializer()
        session.run(init)
        print("W1 = ", str(parameters["W1"].eval()[1, 1, 1]))
        print("W2 = ", str(parameters["W2"].eval()[1, 1, 1]))

    # forward_propagation
    tf.reset_default_graph()
    with tf.Session() as session:
        np.random.seed(1)
        X, Y = create_placeholders(64, 64, 3, 6)
        parameters = initialize_parameters()
        Z3 = forward_propagation(X, parameters)
        init = tf.global_variables_initializer()
        session.run(init)
        a = session.run(Z3, {X: np.random.randn(2, 64, 64, 3), Y: np.random.randn(2, 6)})
        print("Z3 = ", str(a))

    # forward_propagation
    tf.reset_default_graph()
    with tf.Session() as session:
        np.random.seed(1)
        X, Y = create_placeholders(64, 64, 3, 6)
        parameters = initialize_parameters()
        Z3 = forward_propagation(X, parameters)
        cost = compute_cost(Z3, Y)
        init = tf.global_variables_initializer()
        session.run(init)
        a = session.run(cost, {X: np.random.randn(4, 64, 64, 3), Y: np.random.randn(4, 6)})
        print("cost = ", str(a))

    # test model
    _, _, paramters = model(X_train, Y_train, X_test, Y_test,num_epochs=100)
