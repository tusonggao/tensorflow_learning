from __future__ import print_function

# https://www.jianshu.com/p/e79e534afe34

# 去除警告信息
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import time
import numpy as np
import pandas as pd
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization
from keras.optimizers import RMSprop, SGD

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

print('shaep is ', x_train.shape)

exit(0)

# convert class vectors to binary class matrices
print(y_train[0]) # 5
print('y_train.shape is ', y_train.shape, 'num of classes is ', len(np.unique(y_train)))

num_classes = len(np.unique(y_train))
y_train = keras.utils.to_categorical(y_train, num_classes) # 把 y 变成了 one-hot 的形式
print(y_train[0]) # [ 0.  0.  0.  0.  0.  1.  0.  0.  0.  0.]
y_test = keras.utils.to_categorical(y_test, num_classes)


########################################################################################################

# model = Sequential()
# model.add(Dense(input_dim = 28 * 28, units = 633, activation = 'sigmoid'))
# model.add(Dense(units = 633, activation = 'sigmoid'))
# model.add(Dense(units = 633, activation = 'sigmoid'))
# model.add(Dense(units = 10, activation = 'softmax'))
#
# model.compile(loss = 'mse', optimizer = SGD(lr = 0.1), metrics = ['accuracy'])
# model.fit(x_train, y_train, batch_size = 100, epochs = 30)
# result = model.evaluate(x_test, y_test)
# print('\nTest Acc:', result[1])

# 100, 30  0.2072
# 100, 20  0.2083
# 100, 10  0.1135

########################################################################################################

# model = Sequential()
# model.add(Dense(input_dim = 28 * 28, units = 633, activation = 'relu'))
# model.add(Dense(units = 633, activation = 'relu'))
# model.add(Dense(units = 633, activation = 'relu'))
# model.add(Dense(units = 10, activation = 'softmax'))
#
# model.compile(loss = 'mse', optimizer = SGD(lr = 0.1), metrics = ['accuracy'])
#
# start_t = time.time()
# model.fit(x_train, y_train, batch_size = 100, epochs = 10, verbose=2)
# training_t = time.time()-start_t
#
# start_t = time.time()
# result = model.evaluate(x_test, y_test)
# predict_t = time.time()-start_t
#
# print('\nTest Acc:', result[1], 'training time: ', training_t, 'predict time: ', predict_t)

# 10 epochs GPU
# Test Acc: 0.935 training time:  26.156554698944092 predict time:  0.525395393371582

# 10 epochs GPU
# Test Acc: 0.9379 training time:  86.40457391738892 predict time:  0.8622708320617676

# 20 epochs GPU
# Test Acc: 0.9514 training time:  51.74757432937622 predict time:  0.44518351554870605

# 20 epochs CPU
# Test Acc: 0.952 training time:  178.27082991600037 predict time:  0.8853025436401367


########################################################################################################

# model = Sequential()
# model.add(Dense(633, activation='relu', input_shape=(784,)))
# model.add(Dropout(0.2))
# model.add(Dense(633, activation='relu'))
# model.add(Dropout(0.2))
# model.add(Dense(num_classes, activation='softmax'))
#
# model.summary()  # 打印出模型概况
#
# exit(0)
#
# model.compile(loss='categorical_crossentropy',
#               optimizer=RMSprop(),
#               metrics=['accuracy'])
#
# start_t = time.time()
# history = model.fit(x_train, y_train,
#                     batch_size=100,
#                     epochs=10,
#                     verbose=2, # verbose是屏显模式, 0是不屏显，1是显示一个进度条，2是每个epoch都显示一行数据
#                     validation_data=(x_test, y_test))
# training_t = time.time()-start_t
#
# start_t = time.time()
# score = model.evaluate(x_test, y_test, verbose=0)
# predict_t = time.time()-start_t
#
# print('Test loss:', score[0])
# print('Test accuracy:', score[1], 'training time: ', training_t, 'predict time: ', predict_t)


# 20 epochs GPU  512
# Test loss: 0.11427032506950896
# Test accuracy: 0.9828 training time:  59.45206022262573 predict time:  0.3739945888519287

# 10 epochs GPU
# Test loss: 0.10654922620523608
# Test accuracy: 0.9791 training time:  30.82193112373352 predict time:  0.3749973773956299

# 20 epochs GPU  633
# Test loss: 0.14094409557737655
# Test accuracy: 0.9821 training time:  63.14989733695984 predict time:  0.3810131549835205

# 10 epochs GPU  633
# Test loss: 0.10513570926929587
# Test accuracy: 0.9805 training time:  32.52047562599182 predict time:  0.38000988960266113

# 20 epochs CPU
# Test loss: 0.11199523535712733
# Test accuracy: 0.9835 training time:  145.59512853622437 predict time:  0.5484540462493896

# 10 epochs CPU
# Test loss: 0.09506204784913712
# Test accuracy: 0.9816 training time:  70.47938346862793 predict time:  0.5073494911193848


########################################################################################################
#### TSG model

# model = Sequential()
# model.add(Dense(633, activation='relu', input_shape=(784,)))
# model.add(Dropout(0.2))
# model.add(Dense(633, activation='relu'))
# model.add(Dropout(0.2))
# model.add(Dense(32, activation='relu'))
# model.add(Dense(num_classes, activation='softmax'))
#
# model.summary()  # 打印出模型概况
#
# exit(0)
#
# model.compile(loss='categorical_crossentropy',
#               optimizer=RMSprop(),
#               metrics=['accuracy'])
#
#
# start_t = time.time()
# history = model.fit(x_train, y_train,
#                     batch_size=100,
#                     epochs=10,
#                     verbose=2, # verbose是屏显模式, 0是不屏显，1是显示一个进度条，2是每个epoch都显示一行数据
#                     validation_data=(x_test, y_test))
# training_t = time.time()-start_t
#
#
# start_t = time.time()
# score = model.evaluate(x_test, y_test, verbose=0)
# predict_t = time.time()-start_t
#
#
# print('Test loss:', score[0])
# print('Test accuracy:', score[1], 'training time: ', training_t, 'predict time: ', predict_t)


# 20 epochs GPU  512
# Test loss: 0.11427032506950896
# Test accuracy: 0.9828 training time:  59.45206022262573 predict time:  0.3739945888519287

# 10 epochs GPU
# Test loss: 0.10654922620523608
# Test accuracy: 0.9791 training time:  30.82193112373352 predict time:  0.3749973773956299

# 20 epochs GPU  633
# Test loss: 0.14094409557737655
# Test accuracy: 0.9821 training time:  63.14989733695984 predict time:  0.3810131549835205

# 10 epochs GPU  633
# Test loss: 0.10513570926929587
# Test accuracy: 0.9805 training time:  32.52047562599182 predict time:  0.38000988960266113

# 20 epochs CPU
# Test loss: 0.11199523535712733
# Test accuracy: 0.9835 training time:  145.59512853622437 predict time:  0.5484540462493896

# 10 epochs CPU
# Test loss: 0.09506204784913712
# Test accuracy: 0.9816 training time:  70.47938346862793 predict time:  0.5073494911193848


########################################################################################################

#### TSG model using batch normalizaiton 而不是 droput

model = Sequential()
model.add(Dense(633, activation='relu', input_shape=(784,)))
model.add(BatchNormalization())
model.add(Dense(633, activation='relu'))
model.add(BatchNormalization())
model.add(Dense(32, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))

model.summary()  # 打印出模型概况

exit(0)

model.compile(loss='categorical_crossentropy',
              optimizer=RMSprop(),
              metrics=['accuracy'])


start_t = time.time()
history = model.fit(x_train, y_train,
                    batch_size=100,
                    epochs=10,
                    verbose=2, # verbose是屏显模式, 0是不屏显，1是显示一个进度条，2是每个epoch都显示一行数据
                    validation_data=(x_test, y_test))
training_t = time.time()-start_t


start_t = time.time()
score = model.evaluate(x_test, y_test, verbose=0)
predict_t = time.time()-start_t


print('Test loss:', score[0])
print('Test accuracy:', score[1], 'training time: ', training_t, 'predict time: ', predict_t)

