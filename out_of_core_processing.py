import pickle
import time
import random
import gc
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler


def get_train_idx_set(total_file_name, id_column, train_ratio, random_seed=42):
    print('in get_train_idx_set')
    total_df = pd.read_csv(total_file_name, sep='\t')
    buy_user_id_lst = list(total_df[id_column].unique())
    random.seed(random_seed)
    random.shuffle(buy_user_id_lst)
    train_ids = set(buy_user_id_lst[:int(len(buy_user_id_lst) * train_ratio)])
    test_ids = set(buy_user_id_lst[int(len(buy_user_id_lst) * train_ratio):])

    train_idx_set = set()
    for i, id in enumerate(list(total_df[id_column])):
        if id in train_ids:
            train_idx_set.add(i)
    print('len of train_idx_set is ', len(train_idx_set))
    return train_idx_set


def write_to_file(lines, file_name, store_path):
    if not os.path.isdir(store_path):
        os.makedirs(store_path)
    with open(store_path + '/' + file_name, 'w') as file:
        for line in lines:
            file.write(line + '\n')


def write_column_vals_to_files(v_lst, column, batch_num, store_path):
    file_num = 0
    lines = []
    for v in v_lst:
        lines.append(v)
        if len(lines)==batch_num:
            write_to_file(lines,  gen_file_name(column, file_num), store_path)
            lines = []
            file_num += 1
    if len(lines)>0:
        write_to_file(lines, gen_file_name(column, file_num), store_path)
    return 0


def write_to_train_test_files(v_lst, column, batch_num, train_idx_set, data_path):
    train_v_lst = [v_lst[i] for i in range(len(v_lst)) if i in train_idx_set]
    test_v_lst = [v_lst[i] for i in range(len(v_lst)) if i not in train_idx_set]

    train_store_path, test_store_path = data_path+'/train/', data_path+'/test/'
    write_column_vals_to_files(train_v_lst, column, batch_num, train_store_path)
    write_column_vals_to_files(test_v_lst, column, batch_num, test_store_path)


def gen_file_name(column, file_num, digit_num=6):
    return column + '_' + str(file_num).zfill(digit_num)


def column_generator(file_name, i, sep):
    with open(file_name) as file:
        cnt = 0
        for line in file:
            cnt += 1
            if cnt==1:
                continue
            fea = line.strip('\n').split(sep)[i]
            if fea=='':
                fea = np.nan
            yield fea
        print('i is ', i, 'cnt is ', cnt)

def process_value_column(gen, column, batch_num, store_path, train_idx_set):
    def standardized(v_lst):
        df = pd.DataFrame({'val': v_lst})
        df.fillna(df.mean(), inplace=True)
        v_lst = StandardScaler().fit_transform(df['val'].values.reshape(-1, 1))
        v_lst = v_lst.flatten()
        return list(v_lst)

    v_lst = standardized([float(v) for v in gen])
    v_lst = [str(round(v, 8)) for v in v_lst]
    write_to_train_test_files(v_lst, column, batch_num, train_idx_set, store_path)

    return 0

def process_cate_column(gen, column, batch_num, data_path, train_idx_set):
    def get_v_dict(v_lst):
        v_dict = {}
        for v in v_lst:
            if v in v_dict:
                continue
            else:
                v_dict[v] = len(v_dict)
        return v_dict

    def generate_str(num, total_num):
        str_lst = ['0']*total_num
        str_lst[num] = '1'
        return ','.join(str_lst)

    v_lst = [v for v in gen]
    v_dict = get_v_dict(v_lst)
    v_lst = [generate_str(v_dict[v], len(v_dict)) for v in v_lst]

    write_to_train_test_files(v_lst, column, batch_num, train_idx_set, data_path)

def process_id_column(gen, column, batch_num, data_path, train_idx_set):
    v_lst = [v for v in gen]
    write_to_train_test_files(v_lst, column, batch_num, train_idx_set, data_path)


def out_of_core_split(file_name, cate_features, id_column='buy_user_id', drop_features=[],
                      batch_num=1000, header=True, store_path='./data/', train_ratio=0.95, sep=','):
    train_idx_set = get_train_idx_set(file_name, id_column, train_ratio)

    print('in out_of_core_split')
    features_lst = []
    with open(file_name) as file:
        for line in file:
            features_lst = line.strip().split(sep)
            break
    print('total feature num is ', len(features_lst))
    print('features_lst is ', features_lst)

    for i, col in enumerate(features_lst):
        if col in drop_features:  # 如果为丢弃的特征，则不处理直接跳过
            continue

        print('process col ', col)
        gen = column_generator(file_name, i, sep)
        if col in cate_features:
            process_cate_column(gen, col, batch_num, store_path, train_idx_set)
        elif col==id_column:
            process_id_column(gen, col, batch_num, store_path, train_idx_set)
        else:
            process_value_column(gen, col, batch_num, store_path, train_idx_set)

if __name__=='__main__':
    print('hello world!')

    out_of_core_split('./df_merged_processed',
                      cate_features=['address_code', 'class_code', 'branch_code',
                                     'call_month', 'call_weekday', 'first_payment_type',
                                     'first_origin_type', 'last_payment_type', 'last_origin_type',
                                     'first_order_status', 'last_order_status'],
                      id_column='buy_user_id',
                      drop_features=['creation_date'],
                      batch_num=1000,
                      header=True,
                      store_path='./data/',
                      sep='\t')






