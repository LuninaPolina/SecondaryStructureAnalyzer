from keras.models import Model,Sequential
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.layers import BatchNormalization, TimeDistributed, LSTM
from keras import Input
from keras import optimizers
from keras.callbacks import EarlyStopping
import numpy as np
from data_processing import process_file

def precision(cl, row):
    return row[cl] / sum(row)

def recall(cl, col):
    return col[cl] / sum(col)

arr_length = 3028
input_length = 220
retrain = False

model = Sequential()

model.add(Dropout(0.3, input_shape=(arr_length,)))

model.add(Dense(8194, trainable=retrain, name="my_dense_1"))
model.add(BatchNormalization())
model.add(Activation('relu', name='act_1'))

model.add(Dropout(0.9))

model.add(Dense(2048, trainable=retrain, name="my_dense_2"))
model.add(BatchNormalization())
model.add(Activation('relu', name="act_2"))

model.add(Dropout(0.9))

model.add(Dense(1024, trainable=retrain, name="my_dense_3"))
model.add(BatchNormalization())
model.add(Activation('relu', name="act_3"))

model.add(Dropout(0.9))


model.add(Dense(512, trainable=retrain, name="my_dense_4"))
model.add(BatchNormalization())
model.add(Activation('relu', name="act_4"))

model.add(Dropout(0.75))

model.add(Dense(64, trainable=retrain, name="my_dense_5"))

model.add(BatchNormalization())

# Loading weights by name
#model.load_weights("./data/new_weights.h5", by_name=True)

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

inputs = Input(shape=(None, 220))
x = model2 
x = TimeDistributed(x)(inputs) 


# New layers
x = LSTM(32, return_sequences=True)(x)

x = Dense(16)(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)

x = LSTM(16)(x)

x = Dense(4, activation='relu')(x)

outputs = x
model2 = Model(inputs=inputs, outputs=outputs)

model2.load_weights("./data/trained_weights.h5")

# Get data
X_test, y_test = process_file("./data/test.pkl")

y_pred = model2.predict(X_test, verbose=1)
result = np.argmax(y_pred,axis=1) # Get species from probabilities

a = [0, 0, 0, 0]
b = [0, 0, 0, 0]
f = [0, 0, 0, 0]
p = [0, 0, 0, 0]

for i in range(len(result)):
    res = result[i]
    lbl = list(y_test[i]).index(1)
    if lbl == 0:
        a[res] += 1
    if lbl == 1:
        p[res] += 1
    if lbl == 2:
        f[res] += 1
    if lbl == 3:
        b[res] += 1

print(a)
print(p)
print(f)
print(b)

precs = [
    precision(0, [a[0], p[0], f[0], b[0]]),
    precision(1, [a[1], p[1], f[1], b[1]]),
    precision(2, [a[2], p[2], f[2], b[2]]),
    precision(3, [a[3], p[3], f[3], b[3]])
]

recs = [
    recall(0, a),
    recall(1, p),
    recall(2, f),
    recall(3, b)
]
print('prec(a) =', precs[0])
print('prec(b) =', precs[1])
print('prec(f) =', precs[2])
print('prec(p) =', precs[3])

print('rec(a) =', recs[0])
print('rec(b) =', recs[1])
print('rec(f) =', recs[2])
print('rec(p) =', recs[3])

print(f"mean precision: {np.mean(precs)}")
print(f"mean recall: {np.mean(recs)}")
print(f"mean f-score: {2 * np.mean(precs) * np.mean(recs) / (np.mean(precs) + np.mean(recs))}")
