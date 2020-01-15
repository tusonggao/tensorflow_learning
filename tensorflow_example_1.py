# https://v1.tf.wiki/zh/basic.html
import os
import sys
import pandas as pd
import numpy as np
import tensorflow as tf
tf.enable_eager_execution()

A = tf.constant([[1, 2], [3, 4]])
B = tf.constant([[5, 6], [7, 8]])
C = tf.matmul(A, B)

print(C)

# 定义一个随机数（标量）
random_float = tf.random.uniform(shape=())
print('random_float is ', random_float)

# 定义一个有2个元素的零向量
zero_vector = tf.zeros(shape=(2))
print('zero_vector is ', zero_vector)

# 定义两个2×2的常量矩阵
A = tf.constant([[1., 2.], [3., 4.]])
B = tf.constant([[5., 6.], [7., 8.]])
print('A.shape is ', A.shape)

X_raw = np.array([2013, 2014, 2015, 2016, 2017], dtype=np.float32)
y_raw = np.array([12000, 14000, 15000, 16500, 17500], dtype=np.float32)

X = (X_raw - X_raw.min()) / (X_raw.max() - X_raw.min())
y = (y_raw - y_raw.min()) / (y_raw.max() - y_raw.min())

