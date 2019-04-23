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

img_size = 80
batch_size = 64
epochs = 200
train_size = 8000
valid_size = 1000

data_train ='../../data/train'
data_valid ='../../data/valid'

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


model.compile(loss='categorical_crossentropy',
              optimizer=
              optimizers.Adagrad(lr=0.05),
              metrics=['accuracy'])

train_datagen = ImageDataGenerator()
valid_datagen = ImageDataGenerator()



train_generator = train_datagen.flow_from_directory(
        data_train,  
        target_size=(img_size, img_size),
        batch_size=batch_size,
        class_mode='categorical')  


valid_generator = valid_datagen.flow_from_directory(
        data_valid,
        target_size=(img_size, img_size),
        batch_size=batch_size,
        class_mode='categorical')

csv_logger = CSVLogger('training.log')

model.fit_generator(
    train_generator,
    steps_per_epoch=int(train_size/batch_size) - 1,
    validation_data=valid_generator,
    validation_steps=int(valid_size/batch_size) - 1,
    epochs=epochs,
    verbose=2,
    callbacks=[csv_logger])

model.save_weights('weights.h5')
