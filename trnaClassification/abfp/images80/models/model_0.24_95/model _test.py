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

img_size = 80
p, n, total = 0, 0, 0

data_dir ='../../data/test'
weights = 'weights.h5'

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
model.add(Activation('sigmoid'))

model.add(Dense(4))
model.add(Activation('sigmoid'))

model.load_weights(weights)


def test_cls(cls_name, cls_num):
    files = [f for f in glob.glob(data_dir + '/' + cls_name + '/*.bmp')]
    for f in files:
        global total, p, n
        total += 1
        img = image.load_img(f, target_size=(img_size, img_size))
        img = image.img_to_array(img)
        img = np.expand_dims(img, axis=0)
        pred = model.predict_classes(img)
        if pred == [cls_num]:
            p += 1
        else:
            n += 1

test_cls('a', 0)
test_cls('b', 1)
test_cls('f', 2)
test_cls('p', 3)

print("Total: ", total)
print("Positive: ", p, "\r\nNegative: ", n)
