from keras.models import Model,Sequential
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.layers import BatchNormalization, TimeDistributed, LSTM
from keras import Input
from keras import optimizers
from keras.callbacks import EarlyStopping
from data_processing import process_file

# Old model with few layer dropped
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
model.load_weights("./data/new_weights.h5", by_name=True)

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

model2.load_weights("./data/weights.h5")
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


batch_size = 64
opt = optimizers.Adam()

# Monitor to hold best weights
early_stopping_monitor = EarlyStopping(
    monitor='accuracy',
    patience=3,
    restore_best_weights=True
)
# Get data
X_train, y_train = process_file("./data/train.pkl")

# Training
model2.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
model2.fit(X_train, y_train, batch_size=batch_size, epochs=1, callbacks=[early_stopping_monitor])

model2.save_weights("./data/trained_weights.h5")
