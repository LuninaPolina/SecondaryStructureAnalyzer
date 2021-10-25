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
input_length = 220
train_size = 20000
valid_size = 5000
epochs = 100

#specify data paths here
data_train = 'train.csv'
data_valid = 'valid.csv'
db_file = 'ref_db.csv'
weights = 'base_model_weights.h5'

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
model.add(Activation('sigmoid'))

model.add(Dense(1, trainable=False))
model.add(Activation('sigmoid'))

model.load_weights(weights)

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

model2.compile(loss='binary_crossentropy',
              optimizer=
              optimizers.Adagrad(lr=0.05),
              metrics=['accuracy'])


def generate_arrays_from_dir(path, batchsz):
    db = pd.read_csv(db_file)
    while 1:
       with open(path) as f:
            r = csv.reader(f)
            batchCount = 0
            batchX = []
            batchy = []
            for ln in r:
                for i in range(1,len(ln)):
                    if ln[i] == "A": ln[i] = "2"
                    elif ln[i] == "C": ln[i] = "3"
                    elif ln[i] == "G": ln[i] = "5"
                    elif ln[i] == "T": ln[i] = "7"
                    else: ln[i] = "0"
                X = np.array(ln[1:len(ln)],dtype=np.uint32)
                y = 1 if db.loc[db['id'] == int(ln[0][1:])].values[0][2] == 'p' else 0
                batchX.append(np.array(X))
                batchy.append(y)
                batchCount = batchCount + 1
                if batchCount == batchsz:
                    yield (np.array(batchX), np.array(batchy))
                    batchCount = 0
                    batchX = []
                    batchy = []

csv_logger = CSVLogger('training.log')
model2.fit_generator(
    generate_arrays_from_dir(data_train,batch_size),
    steps_per_epoch=int(train_size/batch_size) - 1,
    validation_data=generate_arrays_from_dir(data_valid,batch_size),
    validation_steps=int(valid_size/batch_size) - 1,
    epochs=epochs,
    verbose=2,
    callbacks=[csv_logger])

model2.save_weights('weights.h5')
