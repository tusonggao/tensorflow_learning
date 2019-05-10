# -*- coding: utf-8 -*-
"""
Created on Thu May  9 15:30:06 2019

@author: huzhen
"""

import numpy as np
from tqdm import tqdm

def gen_data_from_file():
    ftrain = open(r'./data/train.txt','w',encoding='utf8')
    ftest = open(r'./data/test.txt','w',encoding='utf8')
    X_y = np.loadtxt('./data/test_hu.txt', delimiter=',')
    for _ in tqdm(X_y):
        _ = ','.join([str(e) for e in _])
        r = np.random.random(1)[0]
        if r > 0.8:
            ftest.write(_+'\n')
        else:
            ftrain.write(_+'\n')
    ftrain.close()
    ftest.close()
gen_data_from_file()

batch_size = 128

def data_generator():
    while True:
        X,Y = [],[]
        with open(r'./data/train.txt','r',encoding='utf8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                line = line.split(',')
                y = int(float(line[-1]))
                x = [float(_) for _ in line[:-1]]
                X.append(x)
                Y.append([y])
                if len(X) == batch_size:
                    yield np.array(X),np.array(Y)
                    X,Y = [],[]
        if X:
            yield np.array(X),np.array(Y)
            X,Y = [],[]
   
def data_generator_test():
    with open(r'./data/test.txt','r',encoding='utf8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            line = line.split(',')
            y = int(float(line[-1]))
            x = [float(_) for _ in line[:-1]]
            yield np.array([x]),y
            
            
from keras.layers import Input,Dense,Dropout
from keras.models import Model
from keras.utils import plot_model
from keras.callbacks import Callback

sen = Input(shape=(1000,),dtype='float32',name='input')
#sen1 = Reshape((times,dims))(sen)
#cnn = Conv1D(dims,3,padding='same',name='cnn1',activation='relu')(sen1)
#cnn = Conv1D(dims,3,padding='same',name='cnn2',activation='relu')(cnn)
##cnn = Conv1D(dims,3,padding='same',name='cnn3',activation='relu')(cnn)
#pool = GlobalMaxPooling1D()(cnn)
#dense = Dense(dims // 2,activation='relu')(pool)
dense = Dense(500,activation='relu')(sen)
dropout = Dropout(0.5)(dense)
output = Dense(1,activation='sigmoid',name='output')(dropout)
model = Model(sen,output)
# plot_model(model,to_file=r'./model_png/model.png',show_shapes=True,show_layer_names=True)
model.compile(
                optimizer = 'adam',
                loss = 'binary_crossentropy',
                metrics = ['accuracy']
              )


class Evaluate(Callback):
    def __init__(self):
        self.highest_acc = 0.
    
    def on_epoch_end(self, epoch, logs=None):
        total = 0
        true = 0
        for x,y in tqdm(data_generator_test(),desc='验证中...'):
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
            print('acc: ',acc,'         ','highest_acc: ',self.highest_acc)
            model.save_weights(r'./model/model.h5')

evaluator = Evaluate()
model.fit_generator(
                      data_generator(),
                      steps_per_epoch = 40000 // batch_size + 1,
                      epochs = 100,
                      callbacks = [evaluator]
                   )