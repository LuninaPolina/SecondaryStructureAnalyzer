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
tf.compat.v1.set_random_seed(1234)

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

model2 = Sequential()

model2.add(Dropout(0.05, input_shape=(input_length,)))

layer = Dense(512)
model2.add(layer)
model2.add(BatchNormalization())
model2.add(Activation('relu'))

layer = Dense(1024)
model2.add(layer)
model2.add(BatchNormalization())
model2.add(Activation('relu'))

layer = Dense(2048)
model2.add(layer)
model2.add(BatchNormalization())
model2.add(Activation('relu'))

layer = Dense(30420)
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

weights = 'model_weights.h5'
model2.load_weights(weights)

df = pd.read_csv(db_file)
classes = sorted(df.phylum.unique())
class2idx = dict(zip(classes, range(len(classes))))
idx2class = dict(zip(range(len(classes)),classes))

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
    for i, row in df.iterrows():
        ln = all_df.loc[str(row.id)]
        ln = list(map(map_values, ln))
        x = np.array(ln, dtype=np.uint32)
        y = np.zeros(len(class2idx))
        y[class2idx[row.phylum]] = 1

        batchX.append(x)
        batchy.append(y)
        batchCount += 1
    return np.stack(batchX), np.stack(batchy)
X,y = get_array_generator('test')

y = np.argmax(y, axis=-1)
y_hat = model2.predict_classes(X, verbose=0)

from sklearn.metrics import accuracy_score,precision_score, recall_score, classification_report

accuracy_score(y, y_hat)
print(classification_report(y, y_hat, target_names=classes))
