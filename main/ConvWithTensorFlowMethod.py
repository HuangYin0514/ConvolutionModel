# -*- coding: utf-8 -*-
# @Time     : 2019/1/15 14:02
# @Author   : HuangYin
# @FileName : ConvWithTensorFlowMethod.py
# @Software : PyCharm

import tensorflow as tf
from tensorflow.python.framework import ops


def create_placeholders(n_H0, n_W0, n_C0, n_y):
    """
        Creates the placeholders for the tensorflow session.

        Arguments:
        n_H0 -- scalar, height of an input image
        n_W0 -- scalar, width of an input image
        n_C0 -- scalar, number of channels of the input
        n_y -- scalar, number of classes

        Returns:
        X -- placeholder for the data input, of shape [None, n_H0, n_W0, n_C0] and dtype "float"
        Y -- placeholder for the input labels, of shape [None, n_y] and dtype "float"
        """

    X = tf.placeholder(tf.float32, [None, n_H0, n_W0, n_C0])
    Y = tf.placeholder(tf.float32, [None, n_y])

    return X, Y


def initialize_parameters():
    """
      Initializes weight parameters to build a neural network with tensorflow. The shapes are:
                          W1 : [4, 4, 3, 8]
                          W2 : [2, 2, 8, 16]
      Returns:
      parameters -- a dictionary of tensors containing W1, W2
      """
    tf.set_random_seed(1)

    W1 = tf.get_variable("W1", [4, 4, 3, 8], initializer=tf.contrib.layers.xavier_initializer(seed=0))
    W2 = tf.get_variable("W2", [2, 2, 8, 16], initializer=tf.contrib.layers.xavier_initializer(seed=0))

    parameters = {"W1": W1,
                  "W2": W2}

    return parameters


def forward_propagation(X, parameters):
    """
      Implements the forward propagation for the model:
      CONV2D -> RELU -> MAXPOOL -> CONV2D -> RELU -> MAXPOOL -> FLATTEN -> FULLYCONNECTED

      Arguments:
      X -- input dataset placeholder, of shape (input size, number of examples)
      parameters -- python dictionary containing your parameters "W1", "W2"
                    the shapes are given in initialize_parameters

      Returns:
      Z3 -- the output of the last LINEAR unit
      """
    W1 = parameters["W1"]
    W2 = parameters["W2"]

    # conv1
    #  stride of 1, padding 'SAME'
    Z1 = tf.nn.conv2d(X, W1, strides=[1, 1, 1, 1], padding="SAME")
    A1 = tf.nn.relu(Z1)
    # MAXPOOL: window 8x8, stride 8, padding 'SAME'
    P1 = tf.nn.max_pool(A1, ksize=[1, 8, 8, 1], strides=[1, 8, 8, 1], padding="SAME")

    # conv2
    # filters W2, stride 1, padding 'SAME'
    Z2 = tf.nn.conv2d(P1, W2, strides=[1, 1, 1, 1], padding="SAME")
    A2 = tf.nn.relu(Z2)
    # MAXPOOL: window 4x4, stride 4, padding 'SAME'
    P2 = tf.nn.max_pool(A2, ksize=[1, 4, 4, 1], strides=[1, 4, 4, 1], padding="SAME")

    # flatten
    P = tf.contrib.layers.flatten(P2)

    # FULLY-CONNECTED without non-linear activation function (not not call softmax).
    # 6 neurons in output layer. Hint: one of the arguments should be "activation_fn=None"
    Z3 = tf.contrib.layers.fully_connected(P, 6, activation_fn=None)

    return Z3
