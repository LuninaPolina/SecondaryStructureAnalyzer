# Load the Drive helper and mount
from google.colab import drive
drive.mount('/content/drive')

cd drive/MyDrive/ML_task/

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
import keras

import pandas as pd
import csv

from keras import optimizers
from PIL import Image
import os
from keras.layers.advanced_activations import PReLU
from keras.callbacks import LearningRateScheduler


import numpy as np
import tensorflow as tf
import random as rn
import struct

#Для одинакового результата при разных запусках.
os.environ['PYTHONHASHSEED'] = '0'
np.random.seed(42)
rn.seed(12345)
session_conf = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
tf.compat.v1.set_random_seed(1234)
sess = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph(), config=session_conf)
tf.compat.v1.keras.backend.set_session(sess)

#specify data paths here
data_train = 'trainBalancedHuge.csv'
data_valid = 'validBalancedHuge.csv'
db_file = 'dbBalancedHuge.csv'
weights = 'weights.h5'

input_length = 220
batch_size = 32
epochs = 150
with open(data_train) as train_file:
  train_size = len(train_file.readlines())
with open(data_valid) as valid_file:
  valid_size = len(valid_file.readlines())

def lr_scheduler(epoch, lr):
    decay_rate = 0.95
    decay_step = 20
    if epoch % decay_step == 0 and epoch != 0:
        print("new lr={}".format(lr * pow(decay_rate, np.floor(epoch / decay_step))))
        return lr * pow(decay_rate, np.floor(epoch / decay_step))
    return lr

model = Sequential()

model.add(Dropout(0.05, input_shape=(input_length,)))

model.add(Dense(512))
model.add(BatchNormalization())
model.add(Activation('relu'))

model.add(Dense(1024))
model.add(BatchNormalization())
model.add(Activation('relu'))

model.add(Dense(2048))
model.add(BatchNormalization())
model.add(Activation('relu'))


model.add(Dense(30420))
model.add(BatchNormalization())
model.add(Activation('relu'))

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
weights_list = model.get_weights()

model2 = Sequential()

model2.add(Dropout(0.05, input_shape=(input_length,)))

layer = Dense(512, weights=[weights_list[0], weights_list[1]])
model2.add(layer)
model2.add(BatchNormalization())
model2.add(Activation('relu'))

layer = Dense(1024, weights=[weights_list[6], weights_list[7]])
model2.add(layer)
model2.add(BatchNormalization())
model2.add(Activation('relu'))

layer = Dense(2048, weights=[weights_list[12], weights_list[13]])
model2.add(layer)
model2.add(BatchNormalization())
model2.add(Activation('relu'))

layer = Dense(30420, weights=[weights_list[18], weights_list[19]])
model2.add(layer)
model2.add(BatchNormalization())
model2.add(Activation('relu'))

model2.add(Dropout(0.3))

layer = Dense(1024, weights=[weights_list[24], weights_list[25]])
model2.add(layer)
model2.add(BatchNormalization())
model2.add(Activation('relu'))

model2.add(Dropout(0.9))

model2.add(Dense(512,  weights=[weights_list[30], weights_list[31]]))
model2.add(BatchNormalization())
model2.add(Activation('relu'))

model2.add(Dropout(0.9))

model2.add(Dense(128,  weights=[weights_list[36], weights_list[37]]))
model2.add(BatchNormalization())
model2.add(Activation('relu'))

model2.add(Dropout(0.75))

model2.add(Dense(64,  weights=[weights_list[42], weights_list[43]]))
model2.add(BatchNormalization())
model2.add(Activation('relu'))

model2.add(Dropout(0.5))

model2.add(Dense(6))
model2.add(Activation('softmax'))


model2.compile(loss='categorical_crossentropy',
              optimizer=
              optimizers.Adagrad(lr=0.05),
              metrics=['accuracy'])

def get_data(path):
  db = pd.read_csv(db_file)
  X_data = []
  y_data = []
  with open(path) as f:
    r = csv.reader(f)
    for line_ind, ln in enumerate(r):            
      id = ln[0]
      if id[0] == '>':
        id = id[1:]

      for i in range(1, len(ln)):
          if ln[i] == "A": ln[i] = "2"
          elif ln[i] == "C": ln[i] = "3"
          elif ln[i] == "G": ln[i] = "5"
          elif ln[i] == "T": ln[i] = "7"
          else: ln[i] = "0"
      X = np.array(ln[1:len(ln)],dtype=np.uint32)
      y = [1, 0, 0, 0, 0, 0]
      if db.loc[db['id'] == int(id)].values[0][2] == 'b':
          y = [0, 1, 0, 0, 0, 0]
      if db.loc[db['id'] == int(id)].values[0][2] == 'f':
          y = [0, 0, 1, 0, 0, 0]
      if db.loc[db['id'] == int(id)].values[0][2] == 'p':
          y = [0, 0, 0, 1, 0, 0]
      if db.loc[db['id'] == int(id)].values[0][2] == 'anim':
          y = [0, 0, 0, 0, 1, 0]
      if db.loc[db['id'] == int(id)].values[0][2] == 'prot':
          y = [0, 0, 0, 0, 0, 1]
      X_data.append(X)
      y_data.append(y)

    return np.array(X_data), np.array(y_data)

train_X, train_y = get_data(data_train)
valid_X, valid_y = get_data(data_valid)

csv_logger = CSVLogger('trainedWeightsHugeB32Down20LR5_300.log')
model2.fit(
    train_X,
    train_y,
    batch_size,
    epochs=300,
    verbose=2,
    initial_epoch=0,
    callbacks=[keras.callbacks.LearningRateScheduler(lr_scheduler), csv_logger],
    validation_data=(valid_X, valid_y))

model2.save_weights('trainedWeightsHugeB32Down20LR5_300.h5')