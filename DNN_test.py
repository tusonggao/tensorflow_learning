import time
import numpy as np
from random import normalvariate  # 正态分布
import matplotlib.pyplot as plt

from tqdm import tqdm_notebook as tqdm

from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from keras.layers import Input,Dense,Dropout
from keras.models import Model
from keras.utils import plot_model
from keras.callbacks import Callback


def gen_data(data_shape):
    file_num = 0
    for i in range(10):
        gen_data((50000, 1000))
        X_data, y_data = gen_data((50000, 1000))
        X_y = np.c_[X_data, y_data]

        n_samples = X_data.shape
        for i in range(0, n_samples, batch_size):
            upper_bound = min(i + batch_size, n_samples)
            ret_x = X_data[perm][i:upper_bound]
            ret_y = y_data[perm][i:upper_bound]
            assert ret_x.shape[0] == ret_y.shape[0]
            yield ret_x, ret_y
    np.random.seed(1001)
    lower, upper = -10, 10
    height, width = data_shape
    X_data = np.random.rand(height, width)*(upper - lower) + lower
    print('X_data.shape is', X_data.shape)
    print('X_data[:, 3] is ', X_data[:, 3])

    col1, col2, col3, col4 = 201, 601, 801, 995

    # y_data = (0.3 * X_data[:, col1] ** 2 + 1.1 * X_data[:, col1] * X_data[:, col2] +
    #           0.6 * X_data[:, col2] ** 2 + 0.7 * X_data[:, col2] * X_data[:, col3]) <= 20

    y_data = (0.3*X_data[:,col1]**2 + 2.3*np.sin(X_data[:,col1]*X_data[:,col2]) +
              1.3*X_data[:,col2]**2 + 1.8*np.cos(X_data[:,col1]*X_data[:,col3]) +
              0.8*X_data[:,col4]**3 + 0.7*np.tan(X_data[:,col3]*X_data[:,col4]) +
              0.5*np.exp(X_data[:,col2]) + 0.8*np.tan(X_data[:,col2]*X_data[:,col4]*2.2) +
              0.75*X_data[:,col1]*X_data[:,col2]*X_data[:,col3] -
              0.52*X_data[:,col1]*X_data[:,col3]*X_data[:,col3] +
              0.38*X_data[:,col4]*X_data[:,col2]*X_data[:,col3] -
              0.45*X_data[:,col1]*X_data[:,col2]*X_data[:,col3]*X_data[:,col4]) <= 60

    print('X_data.shape is', X_data.shape)

    y_data_pos = y_data[y_data == 1]
    y_data_neg = y_data[y_data == 0]

    print('y_data_pos.shape y_data_neg.shape is', y_data_pos.shape, y_data_neg.shape)
    print('X_data.shape is', X_data.shape)

    return X_data, y_data


def gen_data(data_shape):
    np.random.seed(1001)
    lower, upper = -10, 10
    height, width = data_shape
    X_data = np.random.rand(height, width)*(upper - lower) + lower
    print('X_data.shape is', X_data.shape)
    print('X_data[:, 3] is ', X_data[:, 3])

    col1, col2, col3, col4 = 201, 601, 801, 995

    # y_data = (0.3 * X_data[:, col1] ** 2 + 1.1 * X_data[:, col1] * X_data[:, col2] +
    #           0.6 * X_data[:, col2] ** 2 + 0.7 * X_data[:, col2] * X_data[:, col3]) <= 20

    y_data = (0.3*X_data[:,col1]**2 + 2.3*np.sin(X_data[:,col1]*X_data[:,col2]) +
              1.3*X_data[:,col2]**2 + 1.8*np.cos(X_data[:,col1]*X_data[:,col3]) +
              0.8*X_data[:,col4]**3 + 0.7*np.tan(X_data[:,col3]*X_data[:,col4]) +
              0.5*np.exp(X_data[:,col2]) + 0.8*np.tan(X_data[:,col2]*X_data[:,col4]*2.2) +
              0.75*X_data[:,col1]*X_data[:,col2]*X_data[:,col3] -
              0.52*X_data[:,col1]*X_data[:,col3]*X_data[:,col3] +
              0.38*X_data[:,col4]*X_data[:,col2]*X_data[:,col3] -
              0.45*X_data[:,col1]*X_data[:,col2]*X_data[:,col3]*X_data[:,col4]) <= 60

    print('X_data.shape is', X_data.shape)

    y_data_pos = y_data[y_data == 1]
    y_data_neg = y_data[y_data == 0]

    print('y_data_pos.shape y_data_neg.shape is', y_data_pos.shape, y_data_neg.shape)
    print('X_data.shape is', X_data.shape)

    return X_data, y_data

#################################################################################################

def keras_DNN_test(X_train, y_train, X_test, y_test, X_val, y_val):
    sen = Input(shape=(1000,), dtype='float32', name='input')
    # sen1 = Reshape((times,dims))(sen)
    # cnn = Conv1D(dims,3,padding='same',name='cnn1',activation='relu')(sen1)
    # cnn = Conv1D(dims,3,padding='same',name='cnn2',activation='relu')(cnn)
    ##cnn = Conv1D(dims,3,padding='same',name='cnn3',activation='relu')(cnn)
    # pool = GlobalMaxPooling1D()(cnn)
    # dense = Dense(dims // 2,activation='relu')(pool)
    dense = Dense(1500, activation='relu')(sen)
    dropout = Dropout(0.5)(dense)
    dense = Dense(2000, activation='relu')(dropout)
    dropout = Dropout(0.5)(dense)
    dense = Dense(500, activation='relu')(dropout)
    dropout = Dropout(0.5)(dense)
    output = Dense(1, activation='sigmoid', name='output')(dropout)
    model = Model(sen, output)
    # plot_model(model,to_file=r'./model_png/model.png',show_shapes=True,show_layer_names=True)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    def data_generator(X_data, y_data, batch_size=218, epoch=10):
        for i in range(epoch):
            perm = np.random.permutation(X_train.shape[0])
            # X_data = X_data[perm]
            # y_data = y_data[perm]
            n_samples = X_data.shape[0]

            for i in range(0, n_samples, batch_size):
                upper_bound = min(i + batch_size, n_samples)
                ret_x = X_data[perm][i:upper_bound]
                ret_y = y_data[perm][i:upper_bound]
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
    epoch = 200

    evaluator = Evaluate()
    model.fit_generator(
        data_generator(X_train, y_train, batch_size, epoch),
        steps_per_epoch=X_train.shape[0]//batch_size + 1,
        epochs=epoch,
        callbacks=[evaluator]
    )

#################################################################################################

if __name__ == "__main__":
    X_data, y_data = gen_data((50000, 1000))
    # print('y_data[:100]: ', y_data[:100])

    print('X_data.shape is', X_data.shape)
    X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.3, random_state=42)
    X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size=0.5, random_state=42)

    print('X_train.shape is ', X_train.shape, 'X_test.shape is ', X_test.shape, 'X_val.shape is ', X_val.shape)

    keras_DNN_test(X_train, y_train, X_test, y_test, X_val, y_val)


