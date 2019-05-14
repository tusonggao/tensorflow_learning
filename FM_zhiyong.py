'''
Date:20180420
@author: zhaozhiyong
'''

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


from keras.layers import Input,Dense,Dropout
from keras.models import Model
from keras.utils import plot_model
from keras.callbacks import Callback


data_path = 'F:/using_tensorflow/Python-Machine-Learning-Algorithm-3.x-master/Chapter_03 Factorization Machine/code/'


def make_y_data(X_data, use_cols_num=100, seed=1001):
    def sigmoid(x):
        return 1. / (1. + np.exp(-x))

    np.random.seed(seed)
    ratio_array = np.arange(0.11, 2.11, 0.07)
    bias_array = np.arange(-10, 10, 0.3333)

    # op_funcs = {'exp': np.exp, 'sigmoid': sigmoid, 'sin': np.sin, 'cos': np.cos, 'tan': np.tan}

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

def rmse_tsg(y_true, y_pred):
    return np.sqrt(np.mean(np.square(y_true - y_pred)))

# def showData(X_data, y_data):
#     train_data = np.loadtxt(data_path + "train_data.txt")
#     # print('train_data[2] is ', train_data[:, 2])
#     print('train_data.shape is', train_data.shape)
#     train_data_pos = train_data[train_data[:, 2] == 1]
#     train_data_neg = train_data[train_data[:, 2] == 0]
#
#     print('train_data_pos.shape is', train_data_pos.shape)
#     plt.plot(train_data_pos[:, 0], train_data_pos[:, 1], "ro")
#     plt.plot(train_data_neg[:, 0], train_data_neg[:, 1], "bo")
#     plt.show()

def showData(X_data, y_data):
    print('X_data.shape is', X_data.shape)
    X_data_pos = X_data[y_data==1]
    X_data_neg = X_data[y_data==0]

    print('X_data_pos.shape is', X_data_pos.shape)
    plt.plot(X_data_pos[:, 200], X_data_pos[:, 800], "ro")
    plt.plot(X_data_neg[:, 200], X_data_neg[:, 800], "bo")
    plt.show()


# def gen_data_from_file():
#     start_t = time.time()
#     X_y = np.loadtxt('./test_hu.txt', delimiter=',')
#     X, y = X_y[:,:-1], X_y[:, -1]
#     print('X_y.shape is', X_y.shape, 'X.shape is', X.shape, 'y.shape is', y.shape)
#     print('load cost time: ', time.time() - start_t)
#     return X, y


def gen_data_from_file():
    X_y = np.loadtxt('./test_hu.txt', delimiter=',')
    X, y = X_y[:,:-1], X_y[:, -1]
    print('X_y.shape is', X_y.shape, 'X.shape is', X.shape, 'y.shape is', y.shape)
    return X, y


def gen_data_test(data_shape):
    np.random.seed(1001)
    lower, upper = -10, 10
    height, width = data_shape
    X_data = np.random.rand(height, width)*(upper - lower) + lower
    print('X_data.shape is', X_data.shape)
    print('X_data[:, 3] is ', X_data[:, 3])

    col1, col2, col3 = 201, 601, 801
    y_data = (0.3 * X_data[:, col1] ** 2 + 1.1 * X_data[:, col1] * X_data[:, col2] +
              0.6 * X_data[:, col1] ** 2 + 0.7 * X_data[:, col2] * X_data[:, col3]) <= 20

    print('X_data.shape is', X_data.shape)

    y_data_pos = y_data[y_data == 1]
    y_data_neg = y_data[y_data == 0]

    print('y_data_pos.shape y_data_neg.shape is', y_data_pos.shape, y_data_neg.shape)


    X_y = np.c_[X_data, y_data]
    X_y_train = X_y[:50000]
    X_y_test = X_y[50000:]
    start_t = time.time()
    np.savetxt('./test_hu.txt', X_y_train, delimiter=',')
    print('save time cost time: ', time.time() - start_t)

    return X_data, y_data

def gen_data_new(data_shape):
    np.random.seed(1001)
    lower, upper = -10, 10
    height, width = data_shape
    X_data = np.random.rand(height, width)*(upper - lower) + lower
    print('X_data.shape is', X_data.shape)
    print('X_data[:, 3] is ', X_data[:, 3])

    col1, col2, col3 = 111, 222, 333
    # y_data = X_data[:, col1] * X_data[:, col2] * X_data[:, col3] > 0
    y_data = X_data[:, col1] > 0

    print('X_data.shape is', X_data.shape)

    y_data_pos = y_data[y_data == 1]
    y_data_neg = y_data[y_data == 0]

    print('y_data_pos.shape y_data_neg.shape is', y_data_pos.shape, y_data_neg.shape)
    print('X_data.shape is', X_data.shape)

    return X_data, y_data


def gen_data(data_shape):
    # np.random.seed(int(time.time()))
    # np.random.seed(1001)
    lower, upper = -10, 10
    height, width = data_shape
    X_data = np.random.rand(height, width)*(upper - lower) + lower
    print('X_data.shape is', X_data.shape)
    print('X_data[:, 3] is ', X_data[:, 3])

    col1, col2, col3, col4 = 201, 601, 801, 995
    y_data = (0.3*X_data[:, col1]**2 + 2*X_data[:, col1]*X_data[:, col1] +
              1.3*X_data[:, col1]**2) <= 16

    # y_data = (0.3*X_data[:,col1]**2 + 0.3*np.sin(X_data[:,col1]*X_data[:,col2]) +
    #           1.3*X_data[:,col2]**2 + 0.8*np.cos(X_data[:,col1]*X_data[:,col3]) +
    #           0.8*X_data[:,col4]**3 + 0.7*np.tan(X_data[:,col3]*X_data[:,col4]) +
    #           0.5*np.exp(X_data[:,col2]) + 0.8*np.tan(X_data[:,col2]*X_data[:,col4]*2.2) +
    #           0.75*X_data[:,col1]*X_data[:,col2]* X_data[:,col3] -
    #           0.38*X_data[:,col4]*X_data[:,col2]*X_data[:,col3] +
    #           0.45*X_data[:,col1]*X_data[:,col2]*X_data[:,col3]*X_data[:,col4]) <= 40

    # y_data = make_y_data(X_data)

    col1, col2, col3, col4 = 111, 222, 333, 444
    y_data = X_data[:, col1] * X_data[:, col2] * X_data[:, col3] * X_data[:, col4] > 0
    y_data = X_data[:, col1] * X_data[:, col2] * X_data[:, col3] > 0
    # y_data = X_data[:, col1] > 0

    # y_data = np.sin(X_data[:, col1] * X_data[:, col2] * X_data[:, col3]*7)<=0.1

    # fea1 = X_data[:,col1]*X_data[:,col2]* X_data[:,col3]
    # fea2 = X_data[:,col1]*X_data[:,col2]*X_data[:,col3]*X_data[:,col4]
    # X_data = np.c_[X_data, fea1, fea2]

    # y_data = np.random.randint(2, size=height)

    print('X_data.shape is', X_data.shape)

    y_data_pos = y_data[y_data == 1]
    y_data_neg = y_data[y_data == 0]

    print('y_data_pos.shape y_data_neg.shape is', y_data_pos.shape, y_data_neg.shape)
    print('X_data.shape is', X_data.shape)

    return X_data, y_data


#################################################################################################

# def loadDataSet(data):
#     '''导入训练数据
#     input:  data(string)训练数据
#     output: dataMat(list)特征
#             labelMat(list)标签
#     '''
#     dataMat = []
#     labelMat = []
#     fr = open(data)  # 打开文件
#     for line in fr.readlines():
#         lines = line.strip().split("\t")
#         lineArr = []
#
#         for i in range(len(lines) - 1):
#             lineArr.append(float(lines[i]))
#         dataMat.append(lineArr)
#
#         labelMat.append(float(lines[-1]) * 2 - 1)  # 转换成{-1,1}
#         #labelMat.append(float(lines[-1]))
#     fr.close()
#     return dataMat, labelMat
#
# def sigmoid(inx):
#     return 1.0 / (1 + np.exp(-inx))
#
# def initialize_v(n, k):
#     '''初始化交叉项
#     input:  n(int)特征的个数
#             k(int)FM模型的度
#     output: v(mat):交叉项的系数权重
#     '''
#     v = np.mat(np.zeros((n, k)))
#
#     for i in range(n):
#         for j in range(k):
#             # 利用正态分布生成每一个权重
#             v[i, j] = normalvariate(0, 0.2)
#     return v
#
# def stocGradAscent(dataMatrix, classLabels, k, max_iter, alpha):
#     '''利用随机梯度下降法训练FM模型
#     input:  dataMatrix(mat)特征
#             classLabels(mat)标签
#             k(int)v的维数
#             max_iter(int)最大迭代次数
#             alpha(float)学习率
#     output: w0(float),w(mat),v(mat):权重
#     '''
#     m, n = np.shape(dataMatrix)
#     # 1、初始化参数
#     w = np.zeros((n, 1))  # 其中n是特征的个数
#     w0 = 0  # 偏置项
#     v = initialize_v(n, k)  # 初始化V
#
#     # 2、训练
#     for it in range(max_iter):
#         for x in range(m):  # 随机优化，对每一个样本而言的
#             inter_1 = dataMatrix[x] * v
#             inter_2 = np.multiply(dataMatrix[x], dataMatrix[x]) * \
#              np.multiply(v, v)  # multiply对应元素相乘
#             # 完成交叉项
#             interaction = np.sum(np.multiply(inter_1, inter_1) - inter_2) / 2.
#             p = w0 + dataMatrix[x] * w + interaction  # 计算预测的输出
#             loss = sigmoid(classLabels[x] * p[0, 0]) - 1
#
#             w0 = w0 - alpha * loss * classLabels[x]
#             for i in range(n):
#                 if dataMatrix[x, i] != 0:
#                     w[i, 0] = w[i, 0] - alpha * loss * classLabels[x] * dataMatrix[x, i]
#
#                     for j in range(k):
#                         v[i, j] = v[i, j] - alpha * loss * classLabels[x] * \
#                         (dataMatrix[x, i] * inter_1[0, j] -\
#                           v[i, j] * dataMatrix[x, i] * dataMatrix[x, i])
#
#         # 计算损失函数的值
#         if it % 1000 == 0:
#             print("\t------- iter: " + str(it) + " , cost: " + \
#             str(getCost(getPrediction(np.mat(dataMatrix), w0, w, v), classLabels)))
#
#     # 3、返回最终的FM模型的参数
#     return w0, w, v
#
# def getCost(predict, classLabels):
#     '''计算预测准确性
#     input:  predict(list)预测值
#             classLabels(list)标签
#     output: error(float)计算损失函数的值
#     '''
#     m = len(predict)
#     error = 0.0
#     for i in range(m):
#         error -=  np.log(sigmoid(predict[i] * classLabels[i] ))
#     return error
#
# def getPrediction(dataMatrix, w0, w, v):
#     '''得到预测值
#     input:  dataMatrix(mat)特征
#             w(int)常数项权重
#             w0(int)一次项权重
#             v(float)交叉项权重
#     output: result(list)预测的结果
#     '''
#     m = np.shape(dataMatrix)[0]
#     result = []
#     for x in range(m):
#         inter_1 = dataMatrix[x] * v
#         inter_2 = np.multiply(dataMatrix[x], dataMatrix[x]) * \
#          np.multiply(v, v)  # multiply对应元素相乘
#         # 完成交叉项
#         interaction = np.sum(np.multiply(inter_1, inter_1) - inter_2) / 2.
#         p = w0 + dataMatrix[x] * w + interaction  # 计算预测的输出
#         pre = sigmoid(p[0, 0])
#         result.append(pre)
#     return result
#
# def getAccuracy(predict, classLabels):
#     '''计算预测准确性
#     input:  predict(list)预测值
#             classLabels(list)标签
#     output: float(error) / allItem(float)错误率
#     '''
#     m = len(predict)
#     allItem = 0
#     error = 0
#     for i in range(m):
#         allItem += 1
#         if float(predict[i]) < 0.5 and classLabels[i] == 1.0:
#             error += 1
#         elif float(predict[i]) >= 0.5 and classLabels[i] == -1.0:
#             error += 1
#         else:
#             continue
#     return float(error) / allItem
#
# def save_model(file_name, w0, w, v):
#     '''保存训练好的FM模型
#     input:  file_name(string):保存的文件名
#             w0(float):偏置项
#             w(mat):一次项的权重
#             v(mat):交叉项的权重
#     '''
#     f = open(file_name, "w")
#     # 1、保存w0
#     f.write(str(w0) + "\n")
#     # 2、保存一次项的权重
#     w_array = []
#     m = np.shape(w)[0]
#     for i in range(m):
#         w_array.append(str(w[i, 0]))
#     f.write("\t".join(w_array) + "\n")
#     # 3、保存交叉项的权重
#     m1 , n1 = np.shape(v)
#     for i in range(m1):
#         v_tmp = []
#         for j in range(n1):
#             v_tmp.append(str(v[i, j]))
#         f.write("\t".join(v_tmp) + "\n")
#     f.close()

#################################################################################################

def median_mean_guess(X_train, y_train, X_test, y_test):
    median_val = np.median(y_train)
    mean_val = np.mean(y_train)

    median_rmse = np.sqrt(np.square(np.subtract(y_test, median_val)).mean())
    mean_rmse = np.sqrt(np.square(np.subtract(y_test, mean_val)).mean())
    zero_rmse = np.sqrt(np.square(np.subtract(y_test, 0.0)).mean())

    print(f'median_val: {median_val} mean_val:{mean_val} zero_rmse:{zero_rmse}')
    print(f'median_rmse: {median_rmse} mean_rmse:{mean_rmse} zero_rmse:{zero_rmse}')
    return median_rmse, mean_rmse, zero_rmse



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


def FM_tensorflow(X_train, y_train, X_test, y_test, X_val, y_val):
    n, p = X_train.shape

    k = 30
    # k = 50

    x = tf.placeholder('float', [None, p])
    y = tf.placeholder('float', [None, 1])
    w0 = tf.Variable(tf.zeros([1]))
    w = tf.Variable(tf.zeros([p]))
    # v = tf.Variable(tf.random_normal([k, p], mean=0, stddev=0.01))
    v = tf.Variable(tf.random_normal([k, p], mean=1, stddev=0.5))

    linear_terms = tf.add(w0,tf.reduce_sum(tf.multiply(w, x),1,keep_dims=True))
    # pair_interactions = 0.5 * tf.reduce_sum(
    #     tf.subtract(
    #         tf.pow(
    #             tf.matmul(x,tf.transpose(v)),2),
    #         tf.matmul(tf.pow(x,2),tf.transpose(tf.pow(v,2)))
    #     ),axis = 1 , keep_dims=True)

    pair_interactions = 0.5 * tf.reduce_sum(
        tf.add(
            tf.pow(
                tf.matmul(x, tf.transpose(v)), 2),
            tf.matmul(tf.pow(x, 2), tf.transpose(tf.pow(v, 2)))
        ), axis=1, keep_dims=True)


    y_hat = tf.add(linear_terms, pair_interactions)

    lambda_w = tf.constant(0.001, name='lambda_w')
    lambda_v = tf.constant(0.001, name='lambda_v')

    l2_norm = tf.reduce_sum(
        tf.add(
            tf.multiply(lambda_w,tf.pow(w,2)),
            tf.multiply(lambda_v,tf.pow(v,2))
        )
    )

    error = tf.reduce_mean(tf.square(y-y_hat))
    loss = tf.add(error, l2_norm)

    # train_op = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(loss)
    # train_op = tf.train.AdagradOptimizer(learning_rate=0.09).minimize(loss)
    train_op = tf.train.AdamOptimizer(learning_rate=0.01).minimize(loss)

    epochs = 300
    batch_size = 218

    # Launch the graph

    val_RMSE_min = 10**9
    test_rmse_lst, val_rmse_lst, train_rmse_lst = [], [], []
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)

        for epoch in tqdm(range(epochs), unit='epoch'):
            print('epoch is ', epoch)
            perm = np.random.permutation(X_train.shape[0])
            # iterate over batches
            for bX, bY in batcher(X_train[perm], y_train[perm], batch_size):
            # for bX, bY in batcher(x_train[perm], y_train[perm], -1):
                _,t = sess.run([train_op,loss], feed_dict={x: bX.reshape(-1, p), y: bY.reshape(-1, 1)})
                # print(t)

            errors = []
            for bX, bY in batcher(X_train[perm], y_train[perm]):
            # for bX, bY in batcher(x_train[perm], y_train[perm], -1):
                error_rate = sess.run(error, feed_dict={x: bX.reshape(-1, p), y: bY.reshape(-1, 1)})
                errors.append(error_rate)
            train_RMSE = np.sqrt(np.array(errors).mean())
            train_rmse_lst.append(train_RMSE)
            print('train RMSE is ', train_RMSE)

            errors = []
            for bX, bY in batcher(X_test, y_test):
                errors.append(sess.run(error, feed_dict={x: bX.reshape(-1, p), y: bY.reshape(-1, 1)}))
                # print(errors)
            test_RMSE = np.sqrt(np.array(errors).mean())
            test_rmse_lst.append(test_RMSE)
            # MSE = np.array(errors).mean()
            print('test RMSE is ', test_RMSE)

            errors = []
            for bX, bY in batcher(X_val, y_val):
                errors.append(sess.run(error, feed_dict={x: bX.reshape(-1, p), y: bY.reshape(-1, 1)}))
                # print(errors)
            val_RMSE = np.sqrt(np.array(errors).mean())
            val_rmse_lst.append(val_RMSE)
            # MSE = np.array(errors).mean()
            print('val RMSE is ', val_RMSE)

            if val_RMSE_min > val_RMSE:
                val_RMSE_min = val_RMSE

            print('val_RMSE_min is ', val_RMSE_min)


    print('val_RMSE_min is ', val_RMSE_min)
    plt.plot(np.arange(len(train_rmse_lst)), train_rmse_lst, c='red', label='train_error')
    plt.plot(np.arange(len(test_rmse_lst)), test_rmse_lst, c='blue', label='test_error')
    plt.plot(np.arange(len(val_rmse_lst)), val_rmse_lst, c='blue', label='val_error')
    plt.legend(loc='upper left')
    plt.title('Train and Test Error')
    plt.show()

#################################################################################################

def keras_DNN_test(X_train, y_train, X_test, y_test, X_val, y_val):
    sen = Input(shape=(1000,), dtype='float32', name='input')
    # sen1 = Reshape((times,dims))(sen)
    # cnn = Conv1D(dims,3,padding='same',name='cnn1',activation='relu')(sen1)
    # cnn = Conv1D(dims,3,padding='same',name='cnn2',activation='relu')(cnn)
    ##cnn = Conv1D(dims,3,padding='same',name='cnn3',activation='relu')(cnn)
    # pool = GlobalMaxPooling1D()(cnn)
    # dense = Dense(dims // 2,activation='relu')(pool)
    dense = Dense(500, activation='relu')(sen)
    dropout = Dropout(0.5)(dense)
    output = Dense(1, activation='sigmoid', name='output')(dropout)
    model = Model(sen, output)
    # plot_model(model,to_file=r'./model_png/model.png',show_shapes=True,show_layer_names=True)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])



    def data_generator(X_data, y_data, batch_size=218, epoch=10):
        for i in range(epoch):
            perm = np.random.permutation(X_train.shape[0])
            X_data = X_data[perm]
            y_data = y_data[perm]
            n_samples = X_data.shape[0]

            for i in range(0, n_samples, batch_size):
                upper_bound = min(i + batch_size, n_samples)
                ret_x = X_data[i:upper_bound]
                ret_y = y_data[i:upper_bound]
                assert ret_x.shape[0] == ret_y.shape[0]
                yield ret_x, ret_y

    def data_generator_test(X_data, y_data):
        for i in range(X_data.shape[0]):
            # x, y = X_data[i].reshape(1, 1)
            yield X_data[i].reshape(1, 1000), int(y_data[i])

    class Evaluate(Callback):
        def __init__(self):
            self.highest_acc = 0.

        def on_epoch_end(self, epoch, logs=None):
            total = 0
            true = 0
            for x, y in tqdm(data_generator_test(X_test, y_test), desc='验证中...'):
                # print('x.shape y.shape ', x.shape, y)
                y_pred = model.predict(x)[0][0]
                if y_pred >= 0.5:
                    y_pred = 1
                else:
                    y_pred = 0
                if y_pred == y:
                    true += 1
                total += 1
            acc = true / total
            if acc > self.highest_acc:
                self.highest_acc = acc
                print('acc: ', acc, '  ', 'highest_acc: ', self.highest_acc)
                model.save_weights(r'./model/model.h5')

    batch_size = 218
    epoch = 100

    evaluator = Evaluate()
    model.fit_generator(
        data_generator(X_train, y_train, batch_size, epoch),
        steps_per_epoch=X_train.shape[0]//batch_size + 1,
        epochs=epoch,
        callbacks=[evaluator]
    )

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
    # # 1、导入训练数据
    # print("---------- 1.load data ---------")
    # dataTrain, labelTrain = loadDataSet(data_path + "train_data.txt")
    # print("---------- 2.learning ---------")
    # # 2、利用随机梯度训练FM模型
    # w0, w, v = stocGradAscent(np.mat(dataTrain), labelTrain, 3, 10000, 0.01)
    # predict_result = getPrediction(np.mat(dataTrain), w0, w, v)  # 得到训练的准确性
    # print("----------training accuracy: %f" % (1 - getAccuracy(predict_result, labelTrain)))
    # print("---------- 3.save result ---------")
    # # 3、保存训练好的FM模型
    # save_model("weights", w0, w, v)

    # showData()

    X_data, y_data = gen_data((100000, 1000))
    print('y_data[:100]: ', y_data[:100])
    # showData(X_data, y_data)

    print('X_data.shape is', X_data.shape)
    X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.3, random_state=42)
    X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size=0.5, random_state=42)

    print('X_train.shape is ', X_train.shape, 'X_test.shape is ', X_test.shape, 'X_val.shape is ', X_val.shape)

    # y_val_rounded_7 = make_y_data(np.round(X_val, 7))
    #
    # y_val_rounded_6 = make_y_data(np.round(X_val, 6))
    #
    # y_val_rounded_5 = make_y_data(np.round(X_val, 5))
    #
    # print('y_val_rounded_5.sum(): {} y_val_rounded_6.sum(): {} y_val_rounded_7.sum(): {} y_val.sum(): {}'.format(
    #     y_val_rounded_5.sum(),
    #     y_val_rounded_6.sum(),
    #     y_val_rounded_7.sum(),
    #     y_val.sum()))

    # median_mean_guess(X_train, y_train, X_test, y_test)
    # FM_tensorflow(X_train, y_train, X_test, y_test, X_val, y_val)

    # lightGBM_regressor_test(X_train, y_train, X_test, y_test, X_val, y_val)
    lightGBM_classifier_test(X_train, y_train, X_test, y_test, X_val, y_val)

    # keras_DNN_test(X_train, y_train, X_test, y_test, X_val, y_val)

    # gen_data_test((60000, 1000))
    # gen_data_from_file()
