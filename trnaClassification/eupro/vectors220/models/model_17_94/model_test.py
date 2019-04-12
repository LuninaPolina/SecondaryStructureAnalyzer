from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model,Sequential
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K
from keras.preprocessing import image
from keras.layers import BatchNormalization
from keras.callbacks import CSVLogger
from keras.utils import plot_model
from keras.utils.vis_utils import model_to_dot

import pandas as pd
import csv

from keras import optimizers
from PIL import Image
import os
from keras.layers.advanced_activations import PReLU

import numpy as np
import tensorflow as tf
import random as rn
import struct

#Для одинакового результата при разных запусках.
os.environ['PYTHONHASHSEED'] = '0'
np.random.seed(42)
rn.seed(12345)
session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
tf.set_random_seed(1234)
sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
K.set_session(sess)

arr_length = 3028
batch_size = 64
epochs = 100
train_size = 35000
valid_size = 15000

weights = 'weights.h5 '

data = '../data/test.csv'
data_len = 29850


model = Sequential()

model.add(Dropout(0.3, input_shape=(arr_length,)))

model.add(Dense(4096))
model.add(BatchNormalization())
model.add(Activation('relu'))

model.add(Dropout(0.9))

model.add(Dense(2048))
model.add(BatchNormalization())
model.add(Activation('relu'))

model.add(Dropout(0.9))

model.add(Dense(1024))
model.add(BatchNormalization())
model.add(Activation('relu'))

model.add(Dropout(0.9))

model.add(Dense(1024))
model.add(BatchNormalization())
model.add(Activation('relu'))

model.add(Dropout(0.9))

model.add(Dense(512))
model.add(BatchNormalization())
model.add(Activation('relu'))

model.add(Dropout(0.85))

model.add(Dense(256))
model.add(BatchNormalization())
model.add(Activation('relu'))

model.add(Dropout(0.75))

model.add(Dense(256))
model.add(BatchNormalization())
model.add(Activation('relu'))

model.add(Dropout(0.75))

model.add(Dense(32))
model.add(BatchNormalization())
model.add(Activation('relu'))

model.add(Dropout(0.75))

model.add(Dense(8))
model.add(Activation('relu'))

model.add(Dropout(0.5))

model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer=
              optimizers.Adagrad(lr=0.05),
              metrics=['accuracy'])


def generate_arrays_from_dir(path, batchsz):   
    while 1:
       with open(path) as f:
            r = csv.reader(f)
            batchCount = 0
            batchX = []
            batchy = []
            for ln in r:
                X = np.array(list(np.array(ln[1:(len(ln)-1)],dtype=np.uint32).tobytes()))
                y = 1 if db.loc[db['id'] == int(ln[0][1:])].values[0][2] == 'p'
                batchX.append(np.array(X))
                batchy.append(y)
                batchCount = batchCount + 1
                if batchCount == batchsz:
                    yield (np.array(batchX), np.array(batchy))
                    batchCount = 0
                    batchX = []
                    batchy = []

csv_logger = CSVLogger('training.log')

res = model.predict_generator(
         generate_arrays_from_dir(data,batch_size),
         steps = data_len/batch_size)

cnt = 0
for i in range(len(res)):
    if res[i] > 0.5:
        cnt = cnt + 1

print (len(res) - cnt)
print (len(res))
print (cnt)
