import numpy as np
import pandas as pd

import keras
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from keras.datasets import mnist

DATA_DIR = './data/'
NUM_STEPS = 2000
MINIBATCH_SIZE = 100

#################################################################################################
#
# data = input_data.read_data_sets(DATA_DIR, one_hot=True)
#
# x = tf.placeholder(tf.float32, [None, 784])
# W = tf.Variable(tf.zeros([784, 10]))
#
# y_true = tf.placeholder(tf.float32, [None, 10])
# y_pred = tf.matmul(x, W)
#
# cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_pred, labels=y_true))
#
# gd_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
#
# correct_mask = tf.equal(tf.arg_max(y_pred, 1), tf.arg_max(y_true, 1))
# accuracy = tf.reduce_mean(tf.cast(correct_mask, tf.float32))
#
# with tf.Session() as sess:
#     sess.run(tf.global_variables_initializer())
#     for i in range(NUM_STEPS):
#         if i%30:
#             print('i is', i)
#         batch_xs, batch_ys = data.train.next_batch(MINIBATCH_SIZE)
#         sess.run(gd_step, feed_dict={x: batch_xs, y_true:batch_ys})
#     ans = sess.run(accuracy, feed_dict={x: data.test.images,
#                                         y_true: data.test.labels})
#
# print('Accuracy is ', ans)

#################################################################################################
# the data, split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# 由于mnist的输入数据维度是(num, 28, 28)，这里需要把后面的维度直接拼起来变成784维
x_train = x_train.reshape(60000, 784)
x_test = x_test.reshape(10000, 784)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255  # 归一化，所有数值在 0 - 1 之间
x_test /= 255
print(x_train.shape[0], 'train samples') # 60000
print(x_test.shape[0], 'test samples')   # 10000

num_classes = len(np.unique(y_train))
y_train = keras.utils.to_categorical(y_train, num_classes) # 把 y 变成了 one-hot 的形式
print(y_train[0]) # [ 0.  0.  0.  0.  0.  1.  0.  0.  0.  0.]
y_test = keras.utils.to_categorical(y_test, num_classes)

print('x_train shape is ', x_train.shape, 'y_train.shape is ', y_train.shape)

x = tf.placeholder(tf.float32, [None, 784])
# W = tf.Variable(tf.zeros([784, 10]))

W1 = tf.Variable(tf.zeros([784, 150]))
b1 = tf.Variable(0.05, dtype=tf.float32)

W2 = tf.Variable(tf.zeros([150, 10]))
b2 = tf.Variable(0.05, dtype=tf.float32)

y_true = tf.placeholder(tf.float32, [None, 10])
# y_pred = tf.matmul(x, W)


hidden = tf.sigmoid(tf.add(tf.matmul(x, W1),  b1))
y_pred = tf.add(tf.matmul(hidden, W2), b2)


cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_pred, labels=y_true))
gd_step = tf.train.GradientDescentOptimizer(0.15).minimize(cross_entropy)
correct_mask = tf.equal(tf.arg_max(y_pred, 1), tf.arg_max(y_true, 1))
accuracy = tf.reduce_mean(tf.cast(correct_mask, tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(NUM_STEPS):
        batch_idices = np.random.choice(x_train.shape[0], MINIBATCH_SIZE)
        batch_xs, batch_ys = x_train[batch_idices], y_train[batch_idices]
        sess.run(gd_step, feed_dict={x: batch_xs, y_true: batch_ys})

        ans = sess.run(accuracy, feed_dict={x: x_test, y_true: y_test})
        if i%50==0:
            print('i is', i, 'Accuracy is ', ans)

    ans = sess.run(accuracy, feed_dict={x: x_test, y_true: y_test})

print('final Accuracy is ', ans)

###############################################################################################

