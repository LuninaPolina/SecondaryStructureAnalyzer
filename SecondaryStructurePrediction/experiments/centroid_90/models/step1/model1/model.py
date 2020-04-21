from keras.models import Model,Sequential
from keras.models import Sequential
from keras.layers import Conv2D, Input, Activation, BatchNormalization
from keras import backend as K
from keras import layers
from keras.preprocessing import image
from keras.callbacks import CSVLogger
from keras import optimizers
import tensorflow as tf
import numpy as np
import random as rn
from skimage.io import imread
import os
import math

os.environ['PYTHONHASHSEED'] = '0'
np.random.seed(42)
rn.seed(12345)
session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
tf.set_random_seed(1234)
sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
K.set_session(sess)

input_train = '../../data/in/train/'
input_valid = '../../data/in/valid/'
output_train = '../../data/out/train/'
output_valid = '../../data/out/valid/'

img_size = 90
dim = img_size * img_size
batch_size = 8
epochs = 500
train_size = 40817
valid_size = 4535
input_shape = (img_size, img_size, 1)

def load_images(path):
    all_images = []
    for image_path in sorted(os.listdir(path)):
      img = imread(path + image_path , as_gray=True)
      all_images.append(img)
    return np.array(all_images).reshape(len(all_images), img_size, img_size, 1)

def res_unit(inputs, filters, kernel, activ): 
    x = inputs
    x = BatchNormalization()(x)
    for f in filters:
        x = Activation(activ)(x)
        x = Conv2D(f, kernel_size=kernel, padding='same')(x)
    return x

def res_network(units_num, filters, kernel=3, activ = 'relu'):
    inputs = Input(shape=input_shape)
    x = res_unit(inputs, filters, kernel, activ)
    x = layers.add([x, inputs])
    x = Activation(activ)(x)
    for i in range(units_num - 1):
        y = res_unit(x, filters, kernel, activ)
        x = layers.add([x, y])
        x = Activation(activ)(x)
    outputs = x
    
    model = Model(inputs=inputs, outputs=outputs)
    return model

model = res_network(10, [12,10,8,6,1])

coeff = 8
 
def weighted_loss(y_true, y_pred):
    y_true1, y_pred1 = y_true / 255, y_pred / 255
    dif = (1 - K.abs(y_true1 - y_pred1))
    w = (coeff - 1) *y_true1 + 1 
    true_score = K.sum(w, axis = [1,2,3])
    pred_score = K.sum(multiply([dif,w]), axis = [1,2,3])
    loss = K.abs(true_score - pred_score) / true_score
    return K.mean(loss)

def f_mera(y_true, y_pred):
    y_true1, y_pred1 = K.minimum(y_true / 255, 1), K.minimum(y_pred / 255, 1)
    fb = K.cast(K.equal(y_true1, 1),"float32") * K.cast(K.less_equal(y_pred1, 0.25),"float32")
    fw = K.cast(K.equal(y_true1, 0),"float32") * K.cast(K.greater(y_pred1, 0.25),"float32")
    tb = K.cast(K.equal(y_true1, 0),"float32") * K.cast(K.less_equal(y_pred1, 0.25),"float32")
    tw = K.cast(K.equal(y_true1, 1),"float32") * K.cast(K.greater(y_pred1, 0.25),"float32")
    fb = K.sum(fb, axis = [1,2,3]) 
    fw = K.sum(fw, axis = [1,2,3])
    tb = K.sum(tb, axis = [1,2,3])
    tw = K.sum(tw, axis = [1,2,3])
    prec = tw / (tw + fw + 0.0001)
    rec = tw / (tw + fb + 0.0001)
    f_mera = 2 * prec * rec / (prec + rec + 0.0001)
    return K.mean(f_mera)

 
x_train = load_images(input_train)
x_valid = load_images(input_valid)
y_train = load_images(output_train)
y_valid = load_images(output_valid)

model.compile(loss=weighted_loss,
              optimizer=optimizers.Adagrad(lr=0.08),
              metrics = [f_mera])

csv_logger = CSVLogger('training.log')

model.fit(x_train, y_train, 
          validation_data=(x_valid, y_valid), 
          batch_size=batch_size, 
          epochs=epochs, 
          verbose=2,
          shuffle=True,
          callbacks=[csv_logger]) 

model.save_weights('weights.h5')
