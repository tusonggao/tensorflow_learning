# -*- coding: utf-8 -*-
"""
Created on Thu May  9 15:30:06 2019

@author: huzhen
"""

import numpy as np
from tqdm import tqdm
import json

batch_size = 256

def data_generator():
    while True:
        X,Y = [],[]
        with open(r'../data/train.txt','r',encoding='utf8') as f:
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
    with open(r'../data/test.txt','r',encoding='utf8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            line = line.split(',')
            y = int(float(line[-1]))
            x = [float(_) for _ in line[:-1]]
            yield np.array([x]),y
            
            
from keras.layers import Input,Dense,Dropout,Reshape,GlobalAveragePooling1D,Layer
from keras.models import Model
from keras.utils import plot_model
from keras.callbacks import Callback
from keras import backend as K

class GCNN(Layer): # 定义GCNN层，结合残差
    def __init__(self, output_dim=None, residual=False, **kwargs):
        super(GCNN, self).__init__(**kwargs)
        self.output_dim = output_dim
        self.residual = residual
    def build(self, input_shape):
        if self.output_dim == None:
            self.output_dim = input_shape[-1]
        self.kernel = self.add_weight(name='gcnn_kernel',
                                     shape=(3, input_shape[-1],
                                            self.output_dim * 2),
                                     initializer='glorot_uniform',
                                     trainable=True)
    def call(self, x):
        _ = K.conv1d(x, self.kernel, padding='same')
        _ = _[:,:,:self.output_dim] * K.sigmoid(_[:,:,self.output_dim:])
        if self.residual:
            return _ + x
        else:
            return _


sen = Input(shape=(1000,),dtype='float32',name='input')
dense = Dense(1000*10,activation='relu')(sen)
dense = Reshape((50,200))(dense)
#cnn = Conv1D(100,3,padding='same',activation='relu')(dense)
#cnn = Conv1D(100,3,padding='same',activation='relu')(cnn)
#cnn = Conv1D(100,3,padding='same',activation='relu')(cnn)
cnn = GCNN(residual=True)(dense)
cnn = GCNN(residual=True)(cnn)
cnn = GCNN(residual=True)(cnn)
pool = GlobalAveragePooling1D()(cnn)
dropout = Dropout(0.5)(pool)
output = Dense(1,activation='sigmoid',name='output')(dropout)
model = Model(sen,output)
plot_model(model,to_file=r'../model_png/model.png',show_shapes=True,show_layer_names=True)
model.compile(
                optimizer = 'adam',
                loss = 'binary_crossentropy',
                metrics = ['accuracy']
              )
#if 'model.h5' in os.listdir(r'../model/'):
#    model.load_weights(r'../model/model.h5')
#    print('success load model')

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
        print('acc: ',acc)
        if acc > self.highest_acc:
            self.highest_acc = acc
            print('highest_acc: ',self.highest_acc)
            with open(r'../model/acc.json','w',encoding='utf8') as f:
                json.dump(str(self.highest_acc),f,ensure_ascii=False,indent = 4)
            model.save_weights(r'../model/model.h5')

evaluator = Evaluate()
model.fit_generator(
                      data_generator(),
                      steps_per_epoch = 500000 // batch_size + 1,
                      epochs = 200,
                      callbacks = [evaluator]
                   )