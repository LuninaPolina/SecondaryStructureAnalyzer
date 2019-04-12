import numpy as np
from keras.models import Sequential
from keras.layers import Activation, Dropout, Dense
from keras import optimizers
from keras import backend as be
import csv
import matplotlib.pylab as pl

data_dir = '../../data/test.csv'
arr_length = 15625
fp, fn, tp, tn = 0, 0, 0, 0


def generate_arrays(path):
    f = open(path)
    r = csv.reader(f)
    batch_x = []
    batch_y = []
    for ln in r:
        if len(ln) > 0:
            x = np.array(ln[1:(len(ln) - 1)], dtype=np.uint32)
            y = np.array([int(ln[len(ln) - 1] == "p")])
            batch_x.append(np.array(x))
            batch_y.append(y)
    return np.array(batch_x), np.array(batch_y)


model = Sequential()

model.add(Dropout(0.5, input_shape=(arr_length,)))

model.add(Dense(4096))
model.add(Activation('sigmoid'))

model.add(Dropout(0.75))

model.add(Dense(1024))
model.add(Activation('sigmoid'))

model.add(Dropout(0.75))

model.add(Dense(1024))
model.add(Activation('sigmoid'))

model.add(Dropout(0.75))

model.add(Dense(512))
model.add(Activation('sigmoid'))

model.add(Dropout(0.75))

model.add(Dense(512))
model.add(Activation('sigmoid'))

model.add(Dropout(0.75))

model.add(Dense(256))
model.add(Activation('sigmoid'))

model.add(Dropout(0.5))

model.add(Dense(16))
model.add(Activation('sigmoid'))

model.add(Dropout(0.5))

model.add(Dense(1))
model.add(Activation('sigmoid'))

model.load_weights("weights.h5")

data = generate_arrays(data_dir)
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
