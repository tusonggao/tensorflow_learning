# https://datawhatnow.com/feature-importance/
# https://github.com/Microsoft/LightGBM/issues/826

import pickle
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


def DNN_model(train_data, test_data):
    def auc(y_true, y_pred):
        auc = tf.metrics.auc(y_true, y_pred)[1]
        K.get_session().run(tf.local_variables_initializer())
        return auc

    print('in DNN_model')

    train_X, train_y = train_data['X'], train_data['y']
    test_X, test_y = test_data['X'], test_data['y']

    print('train_X.shape is ', train_X.shape, 'train_y.shape is', train_y.shape)
    print('test_X.shape is ', test_X.shape, 'test_y.shape is', test_y.shape)

    init = K.initializers.glorot_uniform(seed=1)

    optimizer_func = K.optimizers.Adam()
    # optimizer_func = K.optimizers.SGD(lr=0.009)

    model = K.models.Sequential()
    # model.add(K.layers.Dense(units=32, input_dim=train_X.shape[1], kernel_initializer=init, activation='relu'))
    model.add(K.layers.Dense(units=633, input_dim=train_X.shape[1], kernel_initializer=init, activation='relu'))
    model.add(K.layers.Dense(units=633, activation='sigmoid'))
    # model.add(K.layers.Dense(units=633, activation='relu'))
    model.add(K.layers.Dense(units=64, activation='relu'))
    # model.add(K.layers.Dense(units=32, activation='sigmoid'))
    # model.add(K.layers.Dense(units=32, activation='relu'))
    # model.add(K.layers.Dense(units=2, kernel_initializer=init, activation='sigmoid'))
    model.add(K.layers.Dense(units=2, activation='sigmoid'))

    model.compile(loss='categorical_crossentropy', optimizer=optimizer_func, metrics=['accuracy'])
    # model.compile(loss='sparse_categorical_crossentropy', optimizer=simple_adam, metrics=['accuracy'])

    # model.compile(loss='categorical_crossentropy', optimizer=simple_adam, metrics=[auc])

    print("Starting training ")
    start_t = time.time()
    h = model.fit(train_X, train_y, batch_size=512, epochs=6, shuffle=True, verbose=1)
    print('Training finished, training cost time: {} \n'.format(time.time()-start_t))

    # 4. 评估模型
    start_t = time.time()
    predict_y = model.predict(test_X)
    print('predict_y.shape is ', predict_y.shape)
    auc_score = roc_auc_score(test_y, predict_y)
    print('auc_score is ', auc_score, 'predict cost time:', time.time() - start_t)
    # eval = model.evaluate(test_x, test_y, verbose=0)
    # print("Evaluation on test data: loss = %0.6f accuracy = %0.2f%% \n" \
    #       % (eval[0], eval[1] * 100))


# def standScaler_tsg(train_X, test_X):
#     merged_df = train_X.append(test_X)
#     merged_df.fillna(merged_df.mean(), inplace=True)
#     merged_df = StandardScaler().fit_transform(merged_df)
#     return merged_df[:len(train_X)], merged_df[len(train_X):]


def standScaler_tsg(train_X, test_X):
    merged_df = train_X.append(test_X)
    merged_df.fillna(merged_df.mean(), inplace=True)
    merged_df = StandardScaler().fit_transform(merged_df)
    return merged_df[:len(train_X)], merged_df[len(train_X):]

def onehot_preprocess(column_dat):
    from numpy import array
    from numpy import argmax
    from sklearn.preprocessing import LabelEncoder
    from sklearn.preprocessing import OneHotEncoder

    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(column_dat)

    onehot_encoder = OneHotEncoder(sparse=False)
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    onehot_encoded = onehot_encoder.fit_transform(integer_encoded)

    return onehot_encoded




# df_merged = get_training_data()
df_merged = get_training_data_quick()

print('\n-------------------------------------\n'
      '     data preprocess finished          \n'
      '---------------------------------------\n')

df_merged_train, df_merged_test = split_by_user_id(df_merged, train_ratio=0.67)
del df_merged

# useless_columns = ['buy_user_id', 'creation_date', 'branch_code']
useless_columns = ['buy_user_id', 'creation_date']
df_merged_train.drop(useless_columns, axis=1, inplace=True)
df_merged_test.drop(useless_columns, axis=1, inplace=True)

print('df_merged_train.dtypes: ',  df_merged_train.dtypes,
      'df_merged_test.dtypes: ', df_merged_test.dtypes)

# print('df_merged_train.shape df_merged_test.shape: ', df_merged_train.shape, df_merged_test.shape)


df_train_y = df_merged_train['y']
df_train_X = df_merged_train.drop(['y'], axis=1)


df_test_y = df_merged_test['y']
df_test_X = df_merged_test.drop(['y'], axis=1)


df_train_X, df_test_X = standScaler_tsg(df_train_X, df_test_X)

df_train_y = to_categorical(df_train_y)
df_test_y = to_categorical(df_test_y)

train_data = {'X': df_train_X, 'y': df_train_y}
test_data = {'X': df_test_X, 'y': df_test_y}

start_t = time.time()
DNN_model(train_data, test_data)
print('DNN_model cost time: ', time.time()-start_t)

print('program ends')