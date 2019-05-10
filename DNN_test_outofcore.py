import time
import numpy as np
from random import normalvariate  # 正态分布
import matplotlib.pyplot as plt
import math

from tqdm import tqdm_notebook as tqdm

from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

import keras as K
from keras.layers import Input,Dense,Dropout
from keras.models import Model
from keras.utils import plot_model, to_categorical
from keras.callbacks import Callback


def save_numpy_to_file(file_name, np_array, digits=6):
    line_str_lst = []
    with open(file_name, 'w') as file:
        for i in range(np_array.shape[0]):
            line_str = ','.join([str(round(val, digits)) for val in np_array[i]])
            line_str_lst.append(line_str)
        file.write('\n'.join(line_str_lst))


def load_numpy_from_file(file_name, delimiter=','):
    X_y = np.loadtxt(file_name, delimiter=delimiter)
    X_data, y_true = X_y[:, :-1], X_y[:, -1]

    col1, col2, col3, col4 = 201, 601, 801, 995
    y_data = make_y_data(X_data)

    print('y_data.sum() is', y_data.sum(), 'y_true.sum() is', y_true.sum(), 'X_y.shape is', X_y.shape)

    # print(X_y.shape)


def gen_data_to_files(batch_size=512):
    file_num = 0
    for i in range(10):
        X_data, y_data = gen_data((batch_size*100, 1000))
        X_y = np.c_[X_data, y_data]
        n_samples = X_data.shape[0]
        for i in range(0, n_samples, batch_size):
            print('i is ', i)
            upper_bound = min(i + batch_size, n_samples)
            save_numpy_to_file('./make_data/' + str(file_num) + '.dat', X_y[i:upper_bound])
            file_num += 1
            print('file_num is', file_num)
    print('generate data file ', file_num)

def make_y_data(X_data, use_cols_num=100, seed=1001):
    def tanh(x):
        s1 = np.exp(x) - np.exp(-x)
        s2 = np.exp(x) + np.exp(-x)
        return s1 / s2

    def sigmoid(x):
        return 1. / (1. + np.exp(-x))

    np.random.seed(seed)
    ratio_array = np.arange(0.11, 2.11, 0.07)
    bias_array = np.arange(-10, 10, 0.3333)

    op_funcs = {'sigmoid': sigmoid, 'sin': np.sin, 'cos': np.cos, 'tan': np.tan}

    y_data = np.ones(X_data.shape[0])
    for i in range(use_cols_num):
        print('i is ', i)
        col_choice = np.random.choice(np.arange(X_data.shape[1]))
        ratio = np.random.choice(ratio_array)
        bias  = np.random.choice(bias_array)
        op_func = op_funcs[np.random.choice([key for key in op_funcs.keys()])]

        y_data += y_data*X_data[:, col_choice]
        y_data = op_func(y_data*ratio + bias)

    y_data = (sigmoid(y_data) < 0.7)
    print('y_data.sum() is', y_data.sum())

    return y_data


def gen_data(data_shape, random_seed=None):
    if random_seed is not None:
        np.random.seed(random_seed)
    lower, upper = -10, 10
    height, width = data_shape
    X_data = np.random.rand(height, width)*(upper - lower) + lower
    print('X_data.shape is', X_data.shape)
    print('X_data[:, 3] is ', X_data[:, 3])

    col1, col2, col3, col4 = 201, 601, 801, 995

    # y_data = (0.3 * X_data[:, col1] ** 2 + 1.1 * X_data[:, col1] * X_data[:, col2] +
    #           0.6 * X_data[:, col2] ** 2 + 0.7 * X_data[:, col2] * X_data[:, col3]) <= 20

    # y_data = (0.3*X_data[:,col1]**2 + 2.3*np.sin(X_data[:,col1]*X_data[:,col2]) +
    #           1.3*X_data[:,col2]**2 + 1.8*np.cos(X_data[:,col1]*X_data[:,col3]) +
    #           0.8*X_data[:,col4]**3 + 0.7*np.tan(X_data[:,col3]*X_data[:,col4]) +
    #           0.5*np.exp(X_data[:,col2]) + 0.8*np.tan(X_data[:,col2]*X_data[:,col4]*2.2) +
    #           0.75*X_data[:,col1]*X_data[:,col2]*X_data[:,col3] -
    #           0.52*X_data[:,col1]*X_data[:,col3]*X_data[:,col3] +
    #           0.38*X_data[:,col4]*X_data[:,col2]*X_data[:,col3] -
    #           0.45*X_data[:,col1]*X_data[:,col2]*X_data[:,col3]*X_data[:,col4]) <= 60

    y_data = make_y_data(X_data)

    print('X_data.shape is', X_data.shape)

    y_data_pos = y_data[y_data == 1]
    y_data_neg = y_data[y_data == 0]

    print('y_data_pos.shape y_data_neg.shape is', y_data_pos.shape, y_data_neg.shape)
    print('X_data.shape is', X_data.shape)

    return X_data, y_data

#################################################################################################

##############################################
    # sen = Input(shape=(1000,), dtype='float32', name='input')
    # dense = Dense(1500, activation='relu')(sen)
    # dropout = Dropout(0.5)(dense)
    # dense = Dense(1000, activation='relu')(dropout)
    # dropout = Dropout(0.5)(dense)
    # output = Dense(2, activation='sigmoid', name='output')(dropout)
    # model = Model(sen, output)
    # model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
################################################

def keras_DNN_test(X_test, y_test):
    sen = Input(shape=(1000,), dtype='float32', name='input')
    # sen1 = Reshape((times,dims))(sen)
    # cnn = Conv1D(dims,3,padding='same',name='cnn1',activation='relu')(sen1)
    # cnn = Conv1D(dims,3,padding='same',name='cnn2',activation='relu')(cnn)
    ##cnn = Conv1D(dims,3,padding='same',name='cnn3',activation='relu')(cnn)
    # pool = GlobalMaxPooling1D()(cnn)
    # dense = Dense(dims // 2,activation='relu')(pool)
    dense = Dense(1500, activation='relu')(sen)
    dropout = Dropout(0.5)(dense)
    # dense = Dense(2000, activation='relu')(dropout)
    # dropout = Dropout(0.5)(dense)
    dense = Dense(1000, activation='relu')(dropout)
    dropout = Dropout(0.5)(dense)

    dense = Dense(300, activation='relu')(dropout)
    dropout = Dropout(0.5)(dense)

    dense = Dense(150, activation='relu')(dropout)
    dropout = Dropout(0.5)(dense)

    dense = Dense(50, activation='relu')(dropout)
    dropout = Dropout(0.5)(dense)

    # output = Dense(1, activation='sigmoid', name='output')(dropout)
    output = Dense(2, activation='sigmoid', name='output')(dropout)
    model = Model(sen, output)
    # plot_model(model,to_file=r'./model_png/model.png',show_shapes=True,show_layer_names=True)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    def generate_instances_from_file(file_nums, data_dir):
        X_y_total = None
        for file_num in file_nums:
            X_y = np.loadtxt(data_dir + str(file_num) + '.dat', delimiter=',')
            if X_y_total is None:
                X_y_total = X_y
            else:
                X_y_total = np.r_[X_y_total, X_y]
        return X_y_total


    def generate_train_batch(file_total_num, data_dir='./make_data/', shuffle_width=3):
        num_arr = np.random.permutation(np.arange(file_total_num))
        nums_splits = np.array_split(num_arr, math.ceil(file_total_num / shuffle_width))

        for file_nums in nums_splits:
            X_y = generate_instances_from_file(file_nums, data_dir)
            np.random.shuffle(X_y)  # 按行打乱顺序
            for X_y_part in np.split(X_y, len(file_nums)):
                yield X_y_part[:, :-1], X_y_part[:, -1]

    max_epochs = 100
    batch_size = 512

    y_test = to_categorical(y_test)
    auc_score_lst = []
    total_batch_num_lst = []
    total_batch_num = 0
    auc_score_best = -999
    test_acc_best = 0.0
    for i in range(max_epochs):
        print('current epoch is ', i)
        batch_num = 0
        for batch_X, batch_y in generate_train_batch(file_total_num=1000):
            batch_num += 1
            total_batch_num += 1
            batch_y = to_categorical(batch_y)
            # print('epoch is ', i,
            #       'batch_num is ', batch_num,
            #       'batch_X.shape is ', batch_X.shape,
            #       'batch_y.shape is ', batch_y.shape)

            model.train_on_batch(batch_X, batch_y)

            if total_batch_num%25==0:
                start_t = time.time()
                predict_y = model.predict(X_test)
                # print('predict_y[:20] is ', predict_y[:20])
                # print('predict_y.shape is ', predict_y.shape)
                auc_score = roc_auc_score(y_test, predict_y)
                print('predict_y.shape ', predict_y.shape, 'y_test.shape is', y_test.shape)
                score = model.evaluate(X_test, y_test, verbose=0)
                test_loss, test_acc = score[0], score[1]
                if test_acc_best < test_acc:
                    test_acc_best = test_acc
                print('epoch:', i, 'total_batch_num:', total_batch_num,
                      'batch_num:', batch_num, 'auc_score:', auc_score,
                      'test_loss:', test_loss, 'test_acc:', test_acc,
                      'test_acc_best :', test_acc_best)

                total_batch_num_lst.append(total_batch_num)
                auc_score_lst.append(auc_score)

                if auc_score_best < auc_score:
                    auc_score_best = auc_score
                    star_t = time.time()
                    model.save('./model/best_model/test_dnn_model.h5', overwrite=True, include_optimizer=True)
                    print('auc_score_best:', auc_score_best, 'model saved cost time:', time.time()-start_t)

            # if total_batch_num>=100:
            #     return


def run_saved_DNN_model():
    start_t = time.time()
    model = K.models.load_model('./model/best_model/test_dnn_model.h5')
    print('load model cost time: ', time.time()-start_t)

    X_test, y_test = gen_data((7500, 1000))
    y_test = to_categorical(y_test)
    print('test_X.shape is', X_test.shape, 'y_test.shape is ', y_test.shape)

    predict_y = model.predict(X_test)
    auc_score_best = roc_auc_score(y_test, predict_y)
    print('auc_score_best is ', auc_score_best)

    score = model.evaluate(X_test, y_test, verbose=0)
    test_loss, test_acc = score[0], score[1]

    print(f'test_loss: {test_loss}  test_acc: {test_acc}')

    return


#################################################################################################

if __name__ == "__main__":
    # gen_data_to_files(batch_size=512)

    load_numpy_from_file('./make_data/105.dat', delimiter=',')

    # X_data, y_data = gen_data((50000, 1000), random_seed=42)
    # print('y_data[:100]: ', y_data[:100])
    #
    # print('X_data.shape is', X_data.shape)
    # X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.3, random_state=42)
    # X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size=0.5, random_state=42)

    # X_val, y_val = gen_data((50000, 1000), random_seed=42)

    X_val, y_val = gen_data((10000, 1000))

    # print('X_val_new.shape is', y_val_new.shape)

    keras_DNN_test(X_val, y_val)

    # run_saved_DNN_model()


    #
    # print('X_train.shape is ', X_train.shape, 'X_test.shape is ', X_test.shape, 'X_val.shape is ', X_val.shape)
    #
    # keras_DNN_test(X_train, y_train, X_test, y_test, X_val, y_val)



