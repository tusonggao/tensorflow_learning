# https://v1.tf.wiki/zh/basic.html
import os
import sys
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import tensorflow as tf
tf.enable_eager_execution()

##########################################################################################

X = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
y = tf.constant([[10.0], [20.0]])

class Linear(tf.keras.Model):
    def __init__(self):
        #super().__init__()
        super(Linear, self).__init__()
        self.dense = tf.keras.layers.Dense(units=1, kernel_initializer=tf.zeros_initializer(),
            bias_initializer=tf.zeros_initializer())

    def call(self, input):
        output = self.dense(input)
        return output

# 以下代码结构与前节类似
model = Linear()
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
#for i in range(100):
for i in range(1):
    with tf.GradientTape() as tape:
        y_pred = model(X)      # 调用模型
        loss = tf.reduce_mean(tf.square(y_pred - y))
    grads = tape.gradient(loss, model.variables)
    print('model.variables is ', model.variables)
    optimizer.apply_gradients(grads_and_vars=zip(grads, model.variables))

print(model.variables)

##########################################################################################################

class DataLoader():
    def __init__(self):
        mnist = tf.contrib.learn.datasets.load_dataset("mnist")
        self.train_data = mnist.train.images                                 # np.array [55000, 784]
        self.train_labels = np.asarray(mnist.train.labels, dtype=np.int32)   # np.array [55000] of int32
        self.eval_data = mnist.test.images                                   # np.array [10000, 784]
        self.eval_labels = np.asarray(mnist.test.labels, dtype=np.int32)     # np.array [10000] of int32

    def get_batch(self, batch_size):
        index = np.random.randint(0, np.shape(self.train_data)[0], batch_size)
        return self.train_data[index, :], self.train_labels[index]

##########################################################################################################

print('prog ends here!')

