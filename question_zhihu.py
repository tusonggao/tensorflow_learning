import time
import numpy as np
from random import normalvariate  # 正态分布
import matplotlib.pyplot as plt

import lightgbm as lgb
import tensorflow as tf

from tqdm import tqdm_notebook as tqdm

from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from keras.layers import Input, Dense, Dropout, BatchNormalization
from keras.models import Model
from keras.utils import plot_model
from keras.optimizers import Adam
from keras.callbacks import Callback


def rmse_tsg(y_true, y_pred):
    return np.sqrt(np.mean(np.square(y_true - y_pred)))

def showData(X_data, y_data):
    print('X_data.shape is', X_data.shape)
    X_data_pos = X_data[y_data==1]
    X_data_neg = X_data[y_data==0]

    print('X_data_pos.shape is', X_data_pos.shape)
    plt.plot(X_data_pos[:, 200], X_data_pos[:, 800], "ro")
    plt.plot(X_data_neg[:, 200], X_data_neg[:, 800], "bo")
    plt.show()


def gen_data(data_shape, random_seed=None):
    if random_seed is not None:
        np.random.seed(random_seed)
    else:
        np.random.seed(int(time.time()))
        # np.random.seed(1001)
    # lower, upper = -10, 10
    lower, upper = -1, 1

    height, width = data_shape
    X_data = np.random.rand(height, width)*(upper - lower) + lower
    # print('X_data.shape is', X_data.shape)
    # print('X_data[:, 3] is ', X_data[:, 3])

    col1, col2, col3, col4 = 111, 222, 333, 444
    # y_data = X_data[:, col1] * X_data[:, col2] * X_data[:, col3] * X_data[:, col4] > 0  # 无法学习到
    # y_data = X_data[:, col1] * X_data[:, col2] * X_data[:, col3] > 0  # 无法学习到
    y_data = (X_data[:, col1] - 0.2) * (X_data[:, col2] - 0.2) * (X_data[:, col3] - 0.2)> 0  #可以学到
    # y_data = (X_data[:, col1] - 0.2) * (X_data[:, col2] - 0.2) > 0  #可以学到
    # y_data = X_data[:, col1] > 0   # 可以学到

    # print('X_data.shape is', X_data.shape)

    y_data_pos = y_data[y_data == 1]
    y_data_neg = y_data[y_data == 0]

    print('y_data_pos.shape y_data_neg.shape is', y_data_pos.shape, y_data_neg.shape)
    # print('X_data.shape is', X_data.shape)

    return X_data, y_data


def batcher(X_, y_=None, batch_size=-1):
    n_samples = X_.shape[0]

    if batch_size == -1:
        batch_size = n_samples
    if batch_size < 1:
       raise ValueError('Parameter batch_size={} is unsupported'.format(batch_size))

    for i in range(0, n_samples, batch_size):
        upper_bound = min(i + batch_size, n_samples)
        ret_x = X_[i:upper_bound]
        ret_y = None
        if y_ is not None:
            ret_y = y_[i:i + batch_size]
            assert ret_x.shape[0]==ret_y.shape[0]
            yield (ret_x, ret_y)


def keras_DNN_test(X_test, y_test):
    sen = Input(shape=(1000,), dtype='float32', name='input')
    # dense = Dense(2000, activation='selu', kernel_initializer='he_uniform')(sen)
    dense = Dense(2000, activation='relu', kernel_initializer='lecun_normal')(sen)
    dense = BatchNormalization()(dense)

    dense = Dense(1000, activation='selu', kernel_initializer='lecun_normal')(dense)
    dense = BatchNormalization()(dense)

    dense = Dense(200, activation='selu', kernel_initializer='lecun_normal')(dense)
    dense = BatchNormalization()(dense)

    # output = Dense(2, activation='sigmoid', name='output')(dense)
    output = Dense(1, activation='sigmoid', name='output')(dense)
    model = Model(sen, output)

    adam = Adam(lr=0.001)
    # model.compile(loss='mean_squared_error', optimizer=adam)
    model.compile(optimizer=adam, loss='binary_crossentropy', metrics=['accuracy'])

    plot_model(model, to_file='./model_test.png', show_shapes=True)

    # model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy']

    def generate_train_batch(batch_size=256):
        # X_data, y_data = gen_data((batch_size * 200, 1000), random_seed=10001)
        X_data, y_data = gen_data((batch_size * 200, 1000))

        n_samples = X_data.shape[0]

        # np.random.seed(int(time.time()))
        rnd_idx = np.random.permutation(len(X_data))
        print('rnd_idx[:10] is', rnd_idx[:10])
        n_batches = len(X_data) // batch_size
        for batch_idx in np.array_split(rnd_idx, n_batches):
            X_batch, y_batch = X_data[batch_idx], y_data[batch_idx]
            yield X_batch, y_batch

        del X_data, y_data

        # for i in range(0, n_samples, batch_size):
        #     upper_bound = min(i + batch_size, n_samples)
        #     yield X_data[i:upper_bound], y_data[i:upper_bound]

    max_epochs = 10
    batch_size = 256

    auc_score_lst = []
    total_batch_num_lst = []
    total_batch_num = 0
    auc_score_best = -999
    test_acc_best = 0.0
    for i in range(max_epochs):
        print('current epoch is ', i)
        batch_num = 0
        for batch_X, batch_y in generate_train_batch(batch_size):
            batch_num += 1
            total_batch_num += 1
            model.train_on_batch(batch_X, batch_y)

            if total_batch_num % 30 == 0:
                start_t = time.time()
                predict_y = model.predict(X_test)
                # print('predict_y[:20] is ', predict_y[:20])
                # print('predict_y.shape is ', predict_y.shape)
                auc_score = roc_auc_score(y_test, predict_y)
                # print('predict_y.shape ', predict_y.shape, 'y_test.shape is', y_test.shape)
                train_loss, train_acc = model.evaluate(batch_X, batch_y, verbose=0)
                test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
                if test_acc_best < test_acc:
                    test_acc_best = test_acc
                print('epoch:', i, 'total_batch_num:', total_batch_num,
                      'batch_num:', batch_num, 'auc_score:', auc_score,
                      'train_loss:', train_loss, 'train_acc:', train_acc,
                      'test_loss:', test_loss, 'test_acc:', test_acc,
                      'test_acc_best :', test_acc_best)

                total_batch_num_lst.append(total_batch_num)
                auc_score_lst.append(auc_score)

                if auc_score_best < auc_score:
                    auc_score_best = auc_score
                    star_t = time.time()
                    model.save('./model/test_dnn_model.h5', overwrite=True, include_optimizer=True)
                    print('auc_score_best:', auc_score_best, 'model saved cost time:', time.time() - start_t)

#################################################################################################

def lightGBM_regressor_test(X_train, y_train, X_test, y_test, X_val, y_val):
    print('in lightGBM_regressor_test')

    lgbm_param = {'n_estimators': 5000, 'n_jobs': -1, 'learning_rate': 0.05,
                  'random_state': 42, 'max_depth': 7, 'min_child_samples': 21,
                  'num_leaves': 17, 'subsample': 0.8, 'colsample_bytree': 0.8,
                  'silent': -1, 'verbose': -1}
    lgbm = lgb.LGBMRegressor(**lgbm_param)
    lgbm.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_test, y_test)],
             eval_metric='rmse', verbose=10, early_stopping_rounds=300)

    y_val_predict = lgbm.predict(X_val)
    rmse_val = rmse_tsg(y_val_predict, y_val)
    print('rmse_val is ', rmse_val)
    return rmse_val


def lightGBM_classifier_test(X_train, y_train, X_test, y_test, X_val, y_val):
    print('in lightGBM_classifier_test')

    lgbm = lgb.LGBMClassifier(n_estimators=2500, n_jobs=-1, learning_rate=0.03,
                             random_state=42, max_depth=15, min_child_samples=700,
                             num_leaves=21, subsample=0.8, colsample_bytree=0.6,
                             silent=-1, verbose=-1)

    lgbm.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_test, y_test)],
             eval_metric= 'auc', verbose=100, early_stopping_rounds=300)

    y_predictions = lgbm.predict_proba(X_val)[:,1]
    auc_val = roc_auc_score(y_val, y_predictions)

    y_predictions = lgbm.predict(X_val)
    accuracy_val = accuracy_score(y_val, y_predictions)

    print(f'auc_val: {auc_val}, accuracy_val val: {accuracy_val}')

    return auc_val, accuracy_val

#################################################################################################

if __name__ == "__main__":
    # X_data, y_data = gen_data((100000, 1000))
    # print('y_data[:100]: ', y_data[:100])
    # # showData(X_data, y_data)
    #
    # print('X_data.shape is', X_data.shape)
    # X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.3, random_state=42)
    # X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size=0.5, random_state=42)
    #
    # print('X_train.shape is ', X_train.shape, 'X_test.shape is ', X_test.shape, 'X_val.shape is ', X_val.shape)

    # lightGBM_classifier_test(X_train, y_train, X_test, y_test, X_val, y_val)

    X_val, y_val = gen_data((20000, 1000), random_seed=42)
    keras_DNN_test(X_val, y_val)

