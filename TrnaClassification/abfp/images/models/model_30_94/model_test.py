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
import glob

#Для одинакового результата при разных запусках.
os.environ['PYTHONHASHSEED'] = '0'
np.random.seed(42)
rn.seed(12345)
session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
tf.set_random_seed(1234)
sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
K.set_session(sess)

data ='../../data/test'
weights = 'weights.h5'


img_size = 80

a = [0, 0, 0, 0]
b = [0, 0, 0, 0]
f = [0, 0, 0, 0]
p = [0, 0, 0, 0]

model = Sequential()

if K.image_data_format() == 'channels_first':
 input_shape = (3, img_size, img_size)
else:
 input_shape = (img_size, img_size, 3)

model.add(Conv2D(5, (3, 3), input_shape=input_shape))
model.add(Activation('relu'))

model.add(Flatten())

model.add(Dropout(0.3))

model.add(Dense(1024))
model.add(BatchNormalization())
model.add(Activation('relu'))

model.add(Dropout(0.9))

model.add(Dense(512))
model.add(BatchNormalization())
model.add(Activation('relu'))

model.add(Dropout(0.9))

model.add(Dense(128))
model.add(BatchNormalization())
model.add(Activation('relu'))

model.add(Dropout(0.75))

model.add(Dense(64))
model.add(BatchNormalization())
model.add(Activation('relu'))

model.add(Dropout(0.5))

model.add(Dense(4))
model.add(Activation('softmax'))

model.load_weights(weights)


def precision(cl, row):
    return row[cl] / sum(row)

def recall(cl, col):
    return col[cl] / sum(col)
    
def test_cls(cls_name, cls_num, res):
    files = [f for f in glob.glob(data + '/' + cls_name + '/*.bmp')]
    for f in files:
        img = image.load_img(f, target_size=(img_size, img_size))
        img = image.img_to_array(img)
        img = np.expand_dims(img, axis=0)
        pred = model.predict_classes(img)
        res[pred[0]] += 1
    return res

a = test_cls('a', 0, a)
b = test_cls('b', 1, b)
f = test_cls('f', 2, f)
p = test_cls('p', 3, p)

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
