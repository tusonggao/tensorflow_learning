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

def show_features_importance_bar(features, feature_importance):
    df_feat_importance = pd.DataFrame({
            'column': features,
            'importance': feature_importance,
        }).sort_values(by='importance', ascending=False)
    df_feat_importance.to_csv('./model_output/df_feat_importance.csv', index=0, sep='\t')

    plt.figure(figsize=(25, 6))
    #plt.yscale('log', nonposy='clip')
    plt.bar(range(len(feature_importance)), feature_importance, align='center')
    plt.xticks(range(len(feature_importance)), features, rotation='vertical')
    plt.title('Feature importance')
    plt.ylabel('Importance')
    plt.xlabel('Features')
    plt.tight_layout()
    plt.show()


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

def generate_real_test_df_quick():
    df_test_real = pd.read_csv('../unassigned/data/hive_sql_unassigned_buyuser_output_processed',
                               sep='\t')
    print('df_test_real.shape is ', df_test_real.shape)
    return df_test_real

def generate_real_test_df():
    print('in generate_real_test_df')

    df_test_real = pd.read_csv('../unassigned/data/hive_sql_unassigned_buyuser_output',
                               parse_dates=[1], infer_datetime_format=True, sep='\t', 
                               names=['buy_user_id', 'creation_date'])

    ###----------------------------###
    ###------ 加上F的feature  ----- ###
    ###----------------------------###
    df_frequency = pd.read_csv('../unassigned/data/hive_sql_F_data.csv', parse_dates=[1], infer_datetime_format=True)
    df_test_real = pd.merge(df_test_real, df_frequency, how='left', on=['buy_user_id', 'creation_date'])
    print('df_test_real.shape after add frequency is ', df_test_real.shape)
    del df_frequency

    ###----------------------------###
    ###------ 加上M的feature  -----###
    ###----------------------------###
    df_monetary = pd.read_csv('../unassigned/data/hive_sql_M_data.csv', parse_dates=[1], infer_datetime_format=True)
    df_test_real = pd.merge(df_test_real, df_monetary, how='left', on=['buy_user_id', 'creation_date'])
    print('df_test_real.shape after add monetary is ', df_test_real.shape)
    del df_monetary

    ###---------------------------------###
    ###   加上first order的features      ###
    ###   包含：                         ###
    ###   1. 订单距离电话回访时间的天数    ###
    ###   2. 订单金额                    ###
    ###   3. 订单来源                    ###
    ###   4. 订单支付方式                ###
    ###---------------------------------###
    df_first_order = pd.read_csv('../unassigned/data/hive_sql_first_order_data.csv',
                                 parse_dates=[1, 2], infer_datetime_format=True,
                                 dtype={'first_origin_type': str, 'first_payment_type': str})
    df_first_order['gap_days_first_order'] = (df_first_order['creation_date'] - df_first_order['order_dt']).dt.days
    df_first_order.drop(['order_dt'], axis=1, inplace=True)
    df_test_real = pd.merge(df_test_real, df_first_order, how='left', on=['buy_user_id', 'creation_date'])
    print('df_test_real.shape after add first order is ', df_test_real.shape)
    del df_first_order

    ###---------------------------------###
    ###   加上last order的features       ###
    ###   包含：                         ###
    ###   1. 订单距离电话回访时间的天数    ###
    ###   2. 订单金额                    ###
    ###   3. 订单来源                    ###
    ###   4. 订单支付方式                ###
    ###---------------------------------###
    df_last_order = pd.read_csv('../unassigned/data/hive_sql_last_order_data.csv',
                                parse_dates=[1, 2], infer_datetime_format=True,
                                dtype={'last_origin_type': str, 'last_payment_type': str})
    df_last_order['gap_days_last_order'] = (df_last_order['creation_date'] - df_last_order['order_dt']).dt.days
    df_last_order.drop(['order_dt'], axis=1, inplace=True)
    df_test_real = pd.merge(df_test_real, df_last_order, how='left', on=['buy_user_id', 'creation_date'])
    print('df_test_real.shape after add last order is ', df_test_real.shape)
    del df_last_order


    ###----------------------------###
    ###--- 收货地址省份的feature  --###
    ###----------------------------###
    df_address = pd.read_csv('../unassigned/data/hive_sql_address_data.csv', dtype={'rand_address_code': str})
    df_address.rename(columns={'rand_address_code':'address_code'}, inplace = True)
    # df_address['address_code'] = df_address['rand_address_code'].apply(str)
    # df_address.drop(['rand_address_code'], axis=1, inplace=True)
    df_test_real = pd.merge(df_test_real, df_address, how='left', on=['buy_user_id'])
    print('df_test_real.shape after add address code is ', df_test_real.shape)
    del df_address


    ###----------------------------------------------------------###
    ###------ 加上class_code 和 branch_code的feature -------------###
    ###----------------------------------------------------------###
    df_class_code = pd.read_csv('../unassigned/data/hive_sql_patient_class_data.csv',
                                dtype={'class_code': str, 'branch_code': str})
    df_test_real = pd.merge(df_test_real, df_class_code, how='left', on=['buy_user_id'])
    print('df_test_real.shape after add class_code, branch_code code is ', df_test_real.shape)
    del df_class_code

    ###----------------------------------------------------------###
    ###------ 加上电话回访时间所在的月份的feature -----------------###
    ###----------------------------------------------------------###
    df_test_real['call_month'] = df_test_real['creation_date'].dt.month.apply(str)
    df_test_real['call_weekday'] = df_test_real['creation_date'].dt.weekday.apply(str)
    print('df_test_real.shape after add call_month call_weekday is ', df_test_real.shape)


    ###----------------------------###
    ###--- 收货地址个数的feature  --###
    ###----------------------------###
    df_address_num = pd.read_csv('../unassigned/data/hive_sql_address_num_data.csv')
    df_merged = pd.merge(df_test_real, df_address_num, how='left', on=['buy_user_id'])
    print('df_test_real.shape after add address number feature is ', df_test_real.shape)
    # print('df_test_real.dtypes after add address number feature is ', df_test_real.dtypes)
    del df_address_num

    start_t = time.time()
    df_test_real.to_csv('../unassigned/data/hive_sql_unassigned_buyuser_output_processed',
                        index=False, sep='\t')    
    print('df_test_real store cost time:', time.time()-start_t)

    return df_test_real

def get_training_data_quick():
    print('in get_training_data_quick()')
    start_t = time.time()
    df_merged = pd.read_csv('../data/df_merged_processed', sep='\t')
    print('read df_merged from csv cost time: ', time.time()-start_t,
          'df_merged.shape:', df_merged.shape)
    return df_merged

def get_training_data():
    print('in get_training_data()')
    df_merged = pd.read_csv('../data/hive_sql_merged_instances.csv', parse_dates=[1],
        infer_datetime_format=True, sep='\t', names=['buy_user_id', 'creation_date', 'y'])    

    #抽样做训练集
    sample_num = 800000
    df_merged = df_merged.sample(n=sample_num, random_state=42)
    print('df_merged shape is ', df_merged.shape)
    # print('df_merged dtypes is ', df_merged.dtypes)

    # split_by_user_id(df_merged)

    ###----------------------------###
    ###------ 加上R的feature  -----###
    ###----------------------------###
    # df_recency = pd.read_csv('./data/hive_sql_R_data.csv', parse_dates=[1, 2], infer_datetime_format=True)
    # df_recency = pd.read_csv('./data/hive_sql_R_data.csv')
    # df_recency['creation_date'] = pd.to_datetime(df_recency['creation_date'], 
    #     format='%Y-%m-%d %H:%M:%S', errors='ignore')
    # df_recency['recency_date'] = pd.to_datetime(df_recency['recency_date'], 
    #     format='%Y-%m-%d %H:%M:%S', errors='ignore')

    # df_recency['gap_days'] = (df_recency['creation_date'] - df_recency['recency_date']).dt.days
    # df_merged = pd.merge(df_merged, df_recency, how='left', on=['buy_user_id', 'creation_date'])
    # df_merged.drop(['recency_date'], axis=1, inplace=True)
    # print('df_merged.shape after add R is ', df_merged.shape)
    # print('df_merged.dtypes after add R is ', df_merged.dtypes)

    # df_merged.drop(['gap_days'], axis=1, inplace=True)

    ###----------------------------###
    ###------ 加上F的feature  ----- ###
    ###----------------------------###
    df_frequency = pd.read_csv('../data/hive_sql_F_data.csv', parse_dates=[1], infer_datetime_format=True)
    df_merged = pd.merge(df_merged, df_frequency, how='left', on=['buy_user_id', 'creation_date'])
    print('df_merged.shape after add frequency is ', df_merged.shape)
    # print('df_merged.dtypes after add frequency is ', df_merged.dtypes)
    del df_frequency

    ###----------------------------###
    ###------ 加上M的feature  -----###
    ###----------------------------###
    df_monetary = pd.read_csv('../data/hive_sql_M_data.csv', parse_dates=[1], infer_datetime_format=True)
    df_merged = pd.merge(df_merged, df_monetary, how='left', on=['buy_user_id', 'creation_date'])
    print('df_merged.shape after add monetary is ', df_merged.shape)
    # print('df_merged.dtypes after add monetary is ', df_merged.dtypes)
    del df_monetary

    ###---------------------------------###
    ###   加上first order的features      ###
    ###   包含：                         ###
    ###   1. 订单距离电话回访时间的天数    ###
    ###   2. 订单金额                    ###
    ###   3. 订单来源                    ###
    ###   4. 订单支付方式                ###
    ###---------------------------------###
    df_first_order = pd.read_csv('../data/hive_sql_first_order_data.csv',
                                 parse_dates=[1, 2], infer_datetime_format=True,
                                 dtype={'first_origin_type': str, 'first_payment_type': str})
    df_first_order['gap_days_first_order'] = (df_first_order['creation_date'] - df_first_order['order_dt']).dt.days
    df_first_order.drop(['order_dt'], axis=1, inplace=True)
    df_merged = pd.merge(df_merged, df_first_order, how='left', on=['buy_user_id', 'creation_date'])
    print('df_merged.shape after add first order is ', df_merged.shape)
    # print('df_merged.dtypes after add first order is ', df_merged.dtypes)
    del df_first_order

    ###---------------------------------###
    ###   加上last order的features       ###
    ###   包含：                         ###
    ###   1. 订单距离电话回访时间的天数    ###
    ###   2. 订单金额                    ###
    ###   3. 订单来源                    ###
    ###   4. 订单支付方式                ###
    ###---------------------------------###
    df_last_order = pd.read_csv('../data/hive_sql_last_order_data.csv',
                                parse_dates=[1, 2], infer_datetime_format=True,
                                dtype={'last_origin_type': str, 'last_payment_type': str})
    df_last_order['gap_days_last_order'] = (df_last_order['creation_date'] - df_last_order['order_dt']).dt.days
    df_last_order.drop(['order_dt'], axis=1, inplace=True)
    df_merged = pd.merge(df_merged, df_last_order, how='left', on=['buy_user_id', 'creation_date'])
    print('df_merged.shape after add last order is ', df_merged.shape)
    # print('df_merged.dtypes after add last order is ', df_merged.dtypes)
    del df_last_order

    ###----------------------------###
    ###--- 收货地址省份的feature  --###
    ###----------------------------###
    df_address = pd.read_csv('../data/hive_sql_address_data.csv', dtype={'rand_address_code': str})
    df_address.rename(columns={'rand_address_code':'address_code'}, inplace = True)
    # df_address['address_code'] = df_address['rand_address_code'].apply(str)
    # df_address.drop(['rand_address_code'], axis=1, inplace=True)
    df_merged = pd.merge(df_merged, df_address, how='left', on=['buy_user_id'])
    print('df_merged.shape after add address code is ', df_merged.shape)
    # print('df_merged.dtypes after add address code is ', df_merged.dtypes)
    del df_address

    ###----------------------------------------------------------###
    ###------ 加上class_code 和 branch_code的feature -------------###
    ###----------------------------------------------------------###
    df_class_code = pd.read_csv('../data/hive_sql_patient_class_data.csv',
                                dtype={'class_code': str, 'branch_code': str})
    # df_class_code['class_code'] = df_class_code['class_code'].apply(str)
    # df_class_code['branch_code'] = df_class_code['branch_code'].apply(str)
    df_merged = pd.merge(df_merged, df_class_code, how='left', on=['buy_user_id'])
    print('df_merged.shape after add class_code, branch_code code is ', df_merged.shape)
    # print('df_merged.dtypes after add class_code, branch_code code is ', df_merged.dtypes)
    del df_class_code

    ###----------------------------------------------------------###
    ###------ 加上start_app count的feature -----------------------###
    ###----------------------------------------------------------###
    # df_start_app_cnt = pd.read_csv('./data/hive_sql_startapp_cnt_data.csv')
    # df_start_app_cnt.rename(columns={'cnt':'start_app_cnt'}, inplace = True)
    # df_merged = pd.merge(df_merged, df_start_app_cnt, how='left', on=['buy_user_id', 'creation_date'])
    # print('df_merged.shape after add start_app count, branch_code code is ', df_merged.shape)
    # print('df_merged.dtypes after add start_app count, branch_code code is ', df_merged.dtypes)
    # del df_start_app_cnt

    ###----------------------------------------------------------###
    ###------ 加上电话回访时间所在的月份的feature -----------------###
    ###----------------------------------------------------------###
    df_merged['call_month'] = df_merged['creation_date'].dt.month.apply(str)
    df_merged['call_weekday'] = df_merged['creation_date'].dt.weekday.apply(str)
    print('df_merged.shape after add start_app count, branch_code code is ', df_merged.shape)
    # print('df_merged.dtypes after add start_app count, branch_code code is ', df_merged.dtypes)

    ###----------------------------###
    ###--- 收货地址个数的feature  --###
    ###----------------------------###
    df_address_num = pd.read_csv('../data/hive_sql_address_num_data.csv')
    df_merged = pd.merge(df_merged, df_address_num, how='left', on=['buy_user_id'])
    print('df_merged.shape after add address number feature is ', df_merged.shape)
    # print('df_merged.dtypes after add address number feature is ', df_merged.dtypes)
    del df_address_num

    start_t = time.time()
    df_merged.to_csv('../data/df_merged_processed', index=False, sep='\t')
    print('store df_merged to_csv cost time: ', time.time()-start_t)
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
    simple_adam = K.optimizers.Adam()
    model = K.models.Sequential()
    # model.add(K.layers.Dense(units=32, input_dim=train_X.shape[1], kernel_initializer=init, activation='relu'))
    model.add(K.layers.Dense(units=512, input_dim=train_X.shape[1], kernel_initializer=init, activation='relu'))
    model.add(K.layers.Dense(units=128, activation='sigmoid'))
    model.add(K.layers.Dense(units=64, activation='relu'))
    model.add(K.layers.Dense(units=64, activation='tanh'))
    model.add(K.layers.Dense(units=32, activation='sigmoid'))
    model.add(K.layers.Dense(units=32, activation='relu'))
    model.add(K.layers.Dense(units=2, kernel_initializer=init, activation='sigmoid'))

    model.compile(loss='categorical_crossentropy', optimizer=simple_adam, metrics=['accuracy'])
    # model.compile(loss='sparse_categorical_crossentropy', optimizer=simple_adam, metrics=['accuracy'])

    # model.compile(loss='categorical_crossentropy', optimizer=simple_adam, metrics=[auc])

    b_size = 512
    max_epochs = 25
    print("Starting training ")
    start_t = time.time()
    h = model.fit(train_X, train_y, batch_size = b_size, epochs = max_epochs, shuffle=True, verbose=1)
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

def standScaler_tsg(train_X, test_X):
    merged_df = train_X.append(test_X)
    merged_df.fillna(merged_df.mean(), inplace=True)
    merged_df = StandardScaler().fit_transform(merged_df)

    return merged_df[:len(train_X)], merged_df[len(train_X):]

# df_merged = get_training_data()
df_merged = get_training_data_quick()

print('\n-------------------------------------\n'
      '     data preprocess finished          \n'
      '---------------------------------------\n')

df_merged_train, df_merged_test = split_by_user_id(df_merged, train_ratio=0.67)
del df_merged

useless_columns = ['buy_user_id', 'creation_date', 'branch_code']
df_merged_train.drop(useless_columns, axis=1, inplace=True)
df_merged_test.drop(useless_columns, axis=1, inplace=True)

print('df_merged_train.shape df_merged_test.shape: ', df_merged_train.shape, df_merged_test.shape)


from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

# scaler = MinMaxScaler(feature_range=(0, 1))
scaler = StandardScaler()


df_train_y = df_merged_train['y']
df_train_X = df_merged_train.drop(['y'], axis=1)
# print('df_train_X.dtypes is ', df_train_X.dtypes)
# df_train_X.fillna(df_train_X.mean(), inplace=True)
# # df_train_X.reset_index(drop=True, inplace=True)
# print('check df_train_X exists nan: ', np.isnan(df_train_X.values).any())
# df_train_X = scaler.fit_transform(df_train_X)

df_test_y = df_merged_test['y']
df_test_X = df_merged_test.drop(['y'], axis=1)
# df_test_X.fillna(df_test_X.mean(), inplace=True)
# # df_test_X.reset_index(drop=True, inplace=True)
# print('check df_test_X exists nan: ', np.isnan(df_test_X.values).any())
# df_test_X = scaler.fit_transform(df_test_X)


df_train_X, df_test_X = standScaler_tsg(df_train_X, df_test_X)

df_train_y = to_categorical(df_train_y)
df_test_y = to_categorical(df_test_y)

train_data = {'X': df_train_X, 'y': df_train_y}
test_data = {'X': df_test_X, 'y': df_test_y}

DNN_model(train_data, test_data)

# d_train = lgb.Dataset(df_train_X.values, label=df_train_y.values, feature_name = feature_names,
#                 categorical_feature=['address_code', 'class_code', 'branch_code',
#                 'call_month', 'call_weekday', 'first_payment_type', 'first_origin_type',
#                 'last_payment_type', 'last_origin_type', 'first_order_status', 'last_order_status'])
#
# params = {'learning_rate':0.08, 'boosting_type':'gbdt', 'objective':'binary',
#           'metric':'binary_logloss', 'sub_feature':0.85, 'sub_sample':0.7,
#           'num_leaves':100, 'min_data':400, 'max_depth':13, 'random_state':42}
#
# print('lgb training starts')
# start_t = time.time()
# clf = lgb.train(params, d_train, 500)
# print('lgb training ends, cost time', time.time()-start_t)
#
# start_t = time.time()
# y_pred=clf.predict(df_test_X.values)
# print('y_pred.shape is ', y_pred.shape)
# auc_score = roc_auc_score(df_test_y, y_pred)
# print('auc_score is ', auc_score, 'predict cost time:', time.time()-start_t)


print('program ends')