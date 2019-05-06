import pickle
import math
import time
import random
import gc
import os
import hashlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import xgboost as xgb
import lightgbm as lgb

import tensorflow as tf
import keras as K
from keras.utils import to_categorical

from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import mean_squared_error

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler


random_seed = 42
np.random.seed(random_seed)


current_path = os.path.dirname(os.path.realpath(__file__))
print('current_path is ', current_path)


from out_of_core_processing import out_of_core_split


# out_of_core_split('./df_merged_processed',
#                       cate_features=['address_code', 'class_code', 'branch_code',
#                                      'call_month', 'call_weekday', 'first_payment_type',
#                                      'first_origin_type', 'last_payment_type', 'last_origin_type',
#                                      'first_order_status', 'last_order_status'],
#                       id_column='buy_user_id',
#                       drop_features=['creation_date'],
#                       batch_num=1000,
#                       header=True,
#                       sep='\t')


def split_by_user_id(df_merged, train_ratio=0.67, random_seed=1001):
    print('in split_by_user_id_new:')

    buy_user_id_lst = list(df_merged['buy_user_id'].unique())

    random.seed(random_seed)
    random.shuffle(buy_user_id_lst)

    train_id_lst = buy_user_id_lst[:int(len(buy_user_id_lst)*train_ratio)]
    test_id_lst = buy_user_id_lst[int(len(buy_user_id_lst)*train_ratio):]

    df_merged_train = df_merged[df_merged['buy_user_id'].isin(train_id_lst)]
    df_merged_test = df_merged[df_merged['buy_user_id'].isin(test_id_lst)]
    return df_merged_train, df_merged_test


def compute_density_multiple(y_true, y_predict, threshold=10, by_percentage=True, top=True):
    df = pd.DataFrame({'y_true': y_true, 'y_predict': y_predict})
    df.sort_values(by=['y_predict'], ascending=False, inplace=True)

    density_whole = sum(df['y_true'])/df.shape[0]
    if by_percentage:
        if top:
            df_target = df[:int(threshold*0.01*df.shape[0])]
        else:
            df_target = df[-int(threshold*0.01*df.shape[0]):]
    else:
        if top:
            df_target = df[:threshold]
        else:
            df_target = df[-threshold:]
    density_partial = sum(df_target['y_true'])/df_target.shape[0]
    density_mutiple = density_partial/density_whole
    return density_mutiple


def get_training_data_quick():
    print('in get_training_data_quick()')
    start_t = time.time()
    df_merged = pd.read_csv(current_path+'/df_merged_processed', sep='\t')
    print('read df_merged from csv cost time: ', time.time()-start_t,
          'df_merged.shape:', df_merged.shape)
    return df_merged


def generate_instances_from_file(columns, file_nums, data_dir):
    concated_matrix = None
    for file_num in file_nums:
        matrix = None
        for column in columns:
            file_name = data_dir + '/' + column + '_' + str(file_num).zfill(6)
            part_matrix = np.loadtxt(file_name, delimiter=',')
            if matrix is None:
                matrix = part_matrix
            else:
                matrix = np.c_[matrix, part_matrix]
        if concated_matrix is None:
            concated_matrix = matrix
        else:
            concated_matrix = np.r_[concated_matrix, matrix]
    return concated_matrix


def generate_train_batch(x_columns, file_total_num, data_dir='./data/train/', shuffle_width=3):
    num_arr = np.random.permutation(np.arange(file_total_num))
    nums_splits = np.array_split(num_arr, math.ceil(file_total_num/shuffle_width))

    for file_nums in nums_splits:
        X = generate_instances_from_file(x_columns, file_nums, data_dir)
        y = generate_instances_from_file(['y'], file_nums, data_dir)
        X_y = np.c_[X, y]
        np.random.shuffle(X_y)  # 按行打乱顺序
        for X_y_part in np.split(X_y, len(file_nums)):
            yield X_y_part[:, :-1], X_y_part[:, -1]

# def generate_test_instances(x_columns, file_total_num=800, file_test_num=10):
#     test_file_nums = np.arange(file_total_num)[-file_test_num:]
#     test_X = generate_instances_from_file(x_columns, test_file_nums, data_dir='./data/')
#     test_y = generate_instances_from_file(['y'], test_file_nums, data_dir='./data/')
#     return test_X, test_y

def generate_test_instances(x_columns, file_total_num, data_dir='./data/test/'):
    test_file_nums = np.arange(file_total_num)
    test_X = generate_instances_from_file(x_columns, test_file_nums, data_dir)
    test_y = generate_instances_from_file(['y'], test_file_nums, data_dir)
    return test_X, test_y


def DNN_model():
    def auc(y_true, y_pred):
        auc = tf.metrics.auc(y_true, y_pred)[1]
        K.get_session().run(tf.local_variables_initializer())
        return auc

    print('in DNN_model')

    init = K.initializers.glorot_uniform(seed=1)

    # optimizer_func = K.optimizers.Adam()
    # optimizer_func = K.optimizers.SGD(lr=0.009)

    model = K.models.Sequential()

    ###############################################################################################
    ### old model
    # model.add(K.layers.Dense(units=512, input_dim=742, kernel_initializer=init, activation='relu'))
    # model.add(K.layers.Dense(units=128, activation='sigmoid'))
    # model.add(K.layers.Dense(units=64, activation='relu'))
    # model.add(K.layers.Dense(units=64, activation='tanh'))
    # model.add(K.layers.Dense(units=32, activation='sigmoid'))
    # model.add(K.layers.Dense(units=32, activation='relu'))
    # model.add(K.layers.Dense(units=2, kernel_initializer=init, activation='sigmoid'))

    ###############################################################################################
    ### new model

    learning_rate = 0.01
    # optimizer_func = K.optimizers.Adam(lr=learning_rate, decay=learning_rate/(760*10))
    optimizer_func = K.optimizers.Adam(lr=0.0012)

    # model.add(K.layers.Dropout(0.8))
    model.add(K.layers.Dense(units=512, input_dim=742, kernel_initializer=init, activation='relu',
                             bias_constraint=K.constraints.max_norm(3)))
    model.add(K.layers.Dropout(0.2))
    model.add(K.layers.Dense(units=128, activation='relu', bias_constraint=K.constraints.max_norm(3)))
    model.add(K.layers.Dropout(0.2))
    model.add(K.layers.Dense(units=64, activation='relu', bias_constraint=K.constraints.max_norm(3)))
    model.add(K.layers.Dropout(0.2))
    model.add(K.layers.Dense(units=32, activation='relu', bias_constraint=K.constraints.max_norm(3)))
    model.add(K.layers.Dropout(0.2))
    model.add(K.layers.Dense(units=2, kernel_initializer=init, activation='sigmoid',
                             bias_constraint=K.constraints.max_norm(3)))


    ###############################################################################################
    ### new model
    # model.add(K.layers.Dense(units=512, input_dim=742, kernel_initializer=init, activation='relu'))
    # model.add(K.layers.Dense(units=128, activation='relu'))
    # model.add(K.layers.Dense(units=64, activation='relu'))
    # model.add(K.layers.Dense(units=64, activation='relu'))
    # model.add(K.layers.Dense(units=32, activation='relu'))
    # model.add(K.layers.Dense(units=32, activation='relu'))
    # model.add(K.layers.Dense(units=2, kernel_initializer=init, activation='sigmoid'))

    ###############################################################################################


    model.compile(loss='categorical_crossentropy', optimizer=optimizer_func, metrics=['accuracy'])
    # model.compile(loss='sparse_categorical_crossentropy', optimizer=simple_adam, metrics=['accuracy'])
    # model.compile(loss='categorical_crossentropy', optimizer=simple_adam, metrics=[auc])

    print("Starting training ")
    start_t = time.time()

    x_columns = ['buy_cnt', 'cost_sum', 'first_order_cost',
                 'first_origin_type', 'first_payment_type',
                 'first_order_status', 'gap_days_first_order',
                 'last_order_cost', 'last_origin_type', 'last_payment_type',
                 'last_order_status', 'gap_days_last_order', 'address_code',
                 'class_code', 'branch_code', 'call_month',
                 'call_weekday', 'address_num']

    # file_total_num, file_test_num = 800, 30

    test_X, test_y = generate_test_instances(x_columns, file_total_num=41)
    test_y = to_categorical(test_y)
    print('test_X.shape is', test_X.shape, 'test_y.shape is ', test_y.shape)

    auc_score_best = -1.0
    total_batch_num = 0
    max_epochs = 2
    total_batch_num_lst, auc_score_lst = [], []

    training_start_t = time.time()
    for i in range(max_epochs):
        print('current epoch is ', i)
        batch_num = 0
        for batch_X, batch_y in generate_train_batch(x_columns, file_total_num=760):
            batch_num += 1
            total_batch_num += 1
            print('epoch is ', i,
                  'batch_num is ', batch_num,
                  'batch_X.shape is ', batch_X.shape,
                  'batch_y.shape is ', batch_y.shape)
            batch_y = to_categorical(batch_y)
            model.train_on_batch(batch_X, batch_y)

            if total_batch_num%10==0:
                start_t = time.time()
                predict_y = model.predict(test_X)
                # print('predict_y[:20] is ', predict_y[:20])
                # print('predict_y.shape is ', predict_y.shape)
                auc_score = roc_auc_score(test_y, predict_y)
                score = model.evaluate(test_X, test_y, verbose=0)
                test_loss, test_acc = score[0], score[1]
                print('epoch:', i, 'total_batch_num:', total_batch_num,
                      'batch_num:', batch_num, 'auc_score:', auc_score,
                      'test_loss:', test_loss, 'test_acc:', test_acc,
                      'eval cost time:', time.time()-start_t)

                total_batch_num_lst.append(total_batch_num)
                auc_score_lst.append(auc_score)

                if auc_score_best < auc_score:
                    auc_score_best = auc_score
                    star_t = time.time()
                    model.save('./model/best_model/model.h5', overwrite=True, include_optimizer=True)
                    print('auc_score_best:', auc_score_best, 'model saved cost time:', time.time()-start_t)


    with open('./model/best_model/auc_lst.pkl', 'wb') as file:
        pickle.dump({'total_batch_num_lst': total_batch_num_lst,
                     'auc_score_lst': auc_score_lst}, file)

    with open('./model/best_model/auc_lst.pkl', 'rb') as file:
        obb = pickle.load(file)
        print('pickle load')
        print(obb['total_batch_num_lst'])

    print('auc_score_best is ', auc_score_best)
    print('Training finished, training cost time: {} \n'.format(time.time() - training_start_t))


def run_saved_DNN_model():
    start_t = time.time()
    model = K.models.load_model('./model/best_model/model.h5')
    print('load model cost time: ', time.time()-start_t)

    x_columns = ['buy_cnt', 'cost_sum', 'first_order_cost',
                 'first_origin_type', 'first_payment_type',
                 'first_order_status', 'gap_days_first_order',
                 'last_order_cost', 'last_origin_type', 'last_payment_type',
                 'last_order_status', 'gap_days_last_order', 'address_code',
                 'class_code', 'branch_code', 'call_month',
                 'call_weekday', 'address_num']

    # file_total_num, file_test_num = 800, 30

    test_X, test_y = generate_test_instances(x_columns, file_total_num=41)
    test_y = to_categorical(test_y)
    print('test_X.shape is', test_X.shape, 'test_y.shape is ', test_y.shape)

    predict_y = model.predict(test_X)
    auc_score_best = roc_auc_score(test_y, predict_y)
    print('origin auc_score_best is ', auc_score_best)

    return

    total_batch_num = 0
    max_epochs = 10
    for i in range(max_epochs):
        print('current epoch is ', i)
        batch_num = 0
        for batch_X, batch_y in generate_train_batch(x_columns, file_total_num=760):
            batch_num += 1
            total_batch_num += 1
            print('epoch is ', i,
                  'batch_num is ', batch_num,
                  'batch_X.shape is ', batch_X.shape,
                  'batch_y.shape is ', batch_y.shape)
            batch_y = to_categorical(batch_y)
            model.train_on_batch(batch_X, batch_y)

            if batch_num % 10 == 0:
                start_t = time.time()
                predict_y = model.predict(test_X)
                auc_score = roc_auc_score(test_y, predict_y)
                score = model.evaluate(test_X, test_y, verbose=0)
                test_loss, test_acc = score[0], score[1]
                print('epoch:', i, 'total_batch_num:', total_batch_num,
                      'batch_num:', batch_num, 'auc_score:', auc_score,
                      'test_loss:', test_loss, 'test_acc:', test_acc,
                      'eval cost time:', time.time() - start_t)
                if auc_score_best < auc_score:
                    auc_score_best = auc_score
                    print('model saved, auc_score_best:', auc_score_best)
                    model.save('./model/best_model/model.h5',
                               overwrite=True, include_optimizer=True)

            if total_batch_num > 60:
                return




# df_merged = get_training_data_quick()

# split_file_by_user_id(current_path+'/df_merged_processed')

start_t = time.time()

DNN_model()

# run_saved_DNN_model()

print('DNN_model cost time: ', time.time()-start_t)

print('program ends')