# https://v1.tf.wiki/zh/basic.html
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

