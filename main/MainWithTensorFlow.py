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
