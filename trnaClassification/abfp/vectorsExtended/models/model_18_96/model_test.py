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


data ='../../data/test.csv'
db_file = '../../data/ref_db.csv'
weights = 'weights.h5'

a = [0, 0, 0, 0]
b = [0, 0, 0, 0]
f = [0, 0, 0, 0]
p = [0, 0, 0, 0]

arr_length = 3028
input_length = 220

model = Sequential()

model.add(Dropout(0.3, input_shape=(arr_length,)))

model.add(Dense(8194, trainable=False))
model.add(BatchNormalization())
model.add(Activation('relu'))

model.add(Dropout(0.9))

model.add(Dense(2048, trainable=False))
model.add(BatchNormalization())
model.add(Activation('relu'))

model.add(Dropout(0.9))

model.add(Dense(1024, trainable=False))
model.add(BatchNormalization())
model.add(Activation('relu'))

model.add(Dropout(0.9))


model.add(Dense(512, trainable=False))
model.add(BatchNormalization())
model.add(Activation('relu'))

model.add(Dropout(0.75))

model.add(Dense(64, trainable=False))
model.add(BatchNormalization())
model.add(Activation('relu'))

model.add(Dropout(0.5))

model.add(Dense(4, trainable=False))
model.add(Activation('softmax'))

model2 = Sequential()

model2.add(Dropout(0.05, input_shape=(input_length,)))

model2.add(Dense(512))
model2.add(BatchNormalization())
model2.add(Activation('relu'))

model2.add(Dense(1024))
model2.add(BatchNormalization())
model2.add(Activation('relu'))

model2.add(Dense(2048))
model2.add(BatchNormalization())
model2.add(Activation('relu'))


model2.add(Dense(3028))
model2.add(BatchNormalization())
model2.add(Activation('relu'))


model2.add(model)

model2.load_weights(weights)

def generate_arrays(path):
    db = pd.read_csv(db_file)
    f = open(path)
    r = csv.reader(f)
    batch_x = []
    batch_y = []
    for ln in r:
        if len(ln) > 0:
            for i in range(1,len(ln)):
                if ln[i] == "A": ln[i] = "2"
                elif ln[i] == "C": ln[i] = "3"
                elif ln[i] == "G": ln[i] = "5"
                elif ln[i] == "T": ln[i] = "7"
                else: ln[i] = "0"
            X = np.array(ln[1:len(ln)],dtype=np.uint32)
            y = [1, 0, 0, 0]
            if db.loc[db['id'] == int(ln[0][1:])].values[0][2] == 'b':
                y = [0, 1, 0, 0]
            if db.loc[db['id'] == int(ln[0][1:])].values[0][2] == 'f':
                y = [0, 0, 1, 0]
            if db.loc[db['id'] == int(ln[0][1:])].values[0][2] == 'p':
                y = [0, 0, 0, 1]
            batch_x.append(np.array(X))
            batch_y.append(y)
    return np.array(batch_x), np.array(batch_y)

data = generate_arrays(data)
result = model2.predict_classes(data[0], verbose=0)

def precision(cl, row):
    return row[cl] / sum(row)

def recall(cl, col):
    return col[cl] / sum(col)

for i in range(len(result)):
    res = result[i]
    lbl = list(data[1][i]).index(1)
    if lbl == 0:
        a[res] += 1
    if lbl == 1:
        b[res] += 1
    if lbl == 2:
        f[res] += 1
    if lbl == 3:
        p[res] += 1        
        
print(a)
print(b)
print(f)
print(p)


print('prec(a) =', precision(0, [a[0], b[0], f[0], p[0]]))
print('prec(b) =', precision(1, [a[1], b[1], f[1], p[1]]))
print('prec(f) =', precision(2, [a[2], b[2], f[2], p[2]]))
print('prec(p) =', precision(3, [a[3], b[3], f[3], p[3]]))

print('rec(a) =', recall(0, a))
print('rec(b) =', recall(1, b))
print('rec(f) =', recall(2, f))
print('rec(p) =', recall(3, p))
