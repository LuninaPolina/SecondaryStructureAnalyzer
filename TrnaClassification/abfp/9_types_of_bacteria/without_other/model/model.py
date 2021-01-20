import tensorflow as tf
from tensorflow import keras 
from tensorflow.keras import optimizers
from tensorflow.keras.models import Model ,Sequential
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import Activation, Dropout, Flatten, Dense
from tensorflow.keras import backend as K
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.callbacks import CSVLogger
from tensorflow.keras.utils import plot_model
from tensorflow.keras.layers import PReLU
 
import pandas as pd
import csv
 
# from PIL import Image
import os
 
import numpy as np
import random as rn
import struct



#Для одинакового результата при разных запусках.
os.environ['PYTHONHASHSEED'] = '0'
np.random.seed(42)
rn.seed(12345)
# session_conf = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
tf.compat.v1.set_random_seed(1234)
# sess = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph(), config=session_conf)
# K.set_session(sess)

#specify data paths here
data_train ='train.csv'
data_valid ='valid.csv'
db_file = 'phylums_bact.csv'
weights = 'base_weights.h5'

img_size = 80
input_length = 220
batch_size = 64
epochs = 150
train_size = 32000
valid_size = 4000


def get_array_generator(phase, batch_size=64):
    df = pd.read_csv(db_file)
    classes = sorted(df.phylum.unique())
    class2idx = dict(zip(classes, range(len(classes))))
    idx2class = dict(zip(range(len(classes)),classes))
    all_df = pd.concat([ pd.read_csv(f'{phase}.csv', header=None) for phase in ['train', 'valid', 'test']])
    all_df[0] = all_df[0].apply(lambda x: x[1:])
    all_df = all_df.set_index(0)
    df = df[df.dataset == phase].copy().sample(frac=1)
    def map_values(v):
        if v == "A": return  "2"
        elif v == "C": return  "3"
        elif v == "G": return  "5"
        elif v == "T": return  "7"
        else: return "0"
    batchCount = 0
    batchX = []
    batchy = []
    while 1:
        for i, row in df.iterrows():
            ln = all_df.loc[str(row.id)]
            ln = list(map(map_values, ln))
            x = np.array(ln, dtype=np.uint32)
            y = np.zeros(len(class2idx), dtype=int)
            y[class2idx[row.phylum]] = 1

            batchX.append(x)
            batchy.append(y)
            batchCount += 1
            if batchCount == batch_size:
                yield (np.stack(batchX), np.stack(batchy))
                batchCount = 0
                batchX = []
                batchy = []

X, y = next(iter(get_array_generator('train')))

if K.image_data_format() == 'channels_first':
    input_shape = (3, img_size, img_size)
else:
    input_shape = (img_size, img_size, 3)


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

layer = Dense(4096)
model2.add(layer)
model2.add(BatchNormalization())
model2.add(Activation('relu'))

model2.add(Dropout(0.9))

layer = Dense(1024)
model2.add(layer)
model2.add(BatchNormalization())
model2.add(Activation('relu'))

model2.add(Dropout(0.8))

layer = Dense(256)
model2.add(layer)
model2.add(BatchNormalization())
model2.add(Activation('relu'))

model2.add(Dropout(0.65))

model2.add(Dense(8))
model2.add(Activation('softmax'))


model2.compile(loss='categorical_crossentropy',
              optimizer=optimizers.Adagrad(lr=0.07),
              metrics=['accuracy'])
csv_logger = CSVLogger('training.log')

train_gen = get_array_generator('train', batch_size)
valid_gen = get_array_generator('valid', batch_size)

model2.fit(
    train_gen,
    steps_per_epoch=int(train_size/batch_size) - 1,
    validation_data=valid_gen,
    validation_steps=int(valid_size/batch_size) - 1,
    epochs=epochs,
    verbose=2,
    callbacks=[csv_logger])

model2.save_weights('model_weights.h5')
