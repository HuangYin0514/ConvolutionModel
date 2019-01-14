# -*- coding: utf-8 -*-
# @Time     : 2019/1/13 16:45
# @Author   : HuangYin
# @FileName : main.py
# @Software : PyCharm
import numpy as np
import h5py
import matplotlib.pyplot as plt
from MyMethod import *

if __name__ == '__main__':
    plt.rcParams['figure.figsize'] = (5.0, 4.0)  # set default size of plots
    plt.rcParams['image.interpolation'] = 'nearest'
    plt.rcParams['image.cmap'] = 'gray'

    np.random.seed(1)

    x = np.random.randn(4, 3, 3, 2)
    x_pad = zero_pad(x, 2)
    print("x.shape =", x.shape)
    print("x_pad.shape =", x_pad.shape)
    print("x[1, 1] =", x[1, 1])
    print("x_pad[1, 1] =", x_pad[1, 1])

    fig, axarr = plt.subplots(1, 2)
    axarr[0].set_title('x')
    axarr[0].imshow(x[0, :, :, 0])
    axarr[1].set_title('x_pad')
    axarr[1].imshow(x_pad[0, :, :, 0])

    # plt.show()

    # ----------convolution multiply
    np.random.seed(1)
    a_slice_prev = np.random.randn(4, 4, 3)
    W = np.random.randn(4, 4, 3)
    b = np.random.randn(1, 1, 1)
    Z = conv_single_step(a_slice_prev, W, b)
    print("Z = ", Z)

    # ----------convolution forward
    np.random.seed(1)
    A_prev = np.random.randn(10, 4, 4, 3)
    W = np.random.randn(2, 2, 3, 8)
    b = np.random.randn(1, 1, 1, 8)
    hparameters = {"pad": 2,
                   "stride": 1}
    Z, cache_conv = conv_forward(A_prev, W, b, hparameters)
    print("Z's mean = ", np.mean(Z))
    print("cache_conv[0][1][2][3] = ", cache_conv[0][1][2][3])

    # --------------pooling forward
    np.random.seed(1)
    A_prev = np.random.randn(2, 4, 4, 3)
    hparameters = {"stride": 1,
                   "f": 4}
    A, cache = pool_forward(A_prev, hparameters)
    print("mode = max")
    print("A = ", A)
    print()
    A, cache = pool_forward(A_prev, hparameters, mode="average")
    print("mode = average")
    print("A = ", A)
    print()

    # -------------------convolution backward
    np.random.seed(1)
    A_prev = np.random.randn(10, 4, 4, 3)
    W = np.random.randn(2, 2, 3, 8)
    b = np.random.randn(1, 1, 1, 8)
    hparameters = {"pad": 2, "stride": 1}

    Z, cache_conv = conv_forward(A_prev, W, b, hparameters)
    dA, dW, db = conv_backward(Z, cache_conv)

    print("dA_mean =", np.mean(dA))
    print("dW_mean =", np.mean(dW))
    print("db_mean =", np.mean(db))
