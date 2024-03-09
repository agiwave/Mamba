
import os
import importlib
import tensorflow as tf
import tensorflow.keras.datasets as datasets
from matplotlib import pyplot as plt

#
# 创建数据集
#
def Dataset(**args):
    (x_train, y_train), (x_test, y_test) = datasets.mnist.load_data()

    x_train = x_train / 255.0
    x_test = x_test / 255.0
    x_train = x_train[..., tf.newaxis]
    x_test = x_test[..., tf.newaxis]

    fig = plt.figure()
    for i in range(12):
        plt.subplot(3, 4, i+1)
        plt.tight_layout()
        plt.imshow(x_train[i])
        #plt.title("Labels: {}".format(train_dataset.targets[i]))
        plt.xticks([])
        plt.yticks([])

    plt.show()

    return (x_train, y_train), (x_test, y_test)