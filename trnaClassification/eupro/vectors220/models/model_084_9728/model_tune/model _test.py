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
fp, fn, tp, tn = 0, 0, 0, 0

data ='../../../data/test.csv'
db_file = '../../../data/ref_db.csv'
weights = 'weights.h5'

model = Sequential()

model.add(Dropout(0.3, input_shape=(arr_length,)))

model.add(Dense(8194))
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

model.add(Dense(512))
model.add(BatchNormalization())
model.add(Activation('relu'))

model.add(Dropout(0.75))

model.add(Dense(64))
model.add(Activation('sigmoid'))

model.add(Dense(1))
model.add(Activation('sigmoid'))


model.load_weights(weights)


def generate_arrays(path):
    db = pd.read_csv(db_file)
    f = open(path)
    r = csv.reader(f)
    batch_x = []
    batch_y = []
    for ln in r:
        if len(ln) > 0:
            x = np.array(list(np.array(ln[1:(len(ln))],dtype=np.uint32).tobytes()))
            y = 1 if db.loc[db['id'] == int(ln[0][1:])].values[0][2] == 'p' else 0
            batch_x.append(np.array(x))
            batch_y.append(y)
    return np.array(batch_x), np.array(batch_y)

data = generate_arrays(data)
res = model.predict_classes(data[0], verbose=0)
print("Total: ", len(res))

for i in range(len(res)):
    if res[i] == 1 and data[1][i] == 1:
        tp += 1
    elif res[i] == 1 and data[1][i] == 0:
        fp += 1
    elif res[i] == 0 and data[1][i] == 1:
        fn += 1
    else:
        tn += 1

print("True Positive: ", tp, "\r\nFalse Positive: ", fp, "\r\nTrue Negative: ", tn, "\r\nFalse Negative: ", fn)



