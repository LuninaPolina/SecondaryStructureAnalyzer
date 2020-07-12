from keras.models import Model
from keras.layers import Conv2D, Input, multiply, Activation, BatchNormalization, add 
from keras import backend as K
from keras.callbacks import CSVLogger
from keras.optimizers import Adagrad
import tensorflow as tf
import numpy as np
import random as rn
from skimage.io import imread
import os
from tensorflow.python.ops.math_ops import square
import math
import glob
from PIL import Image


#Для одинакового результата при разных запусках.
os.environ['PYTHONHASHSEED'] = '0'
np.random.seed(42)
rn.seed(12345)
session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
tf.set_random_seed(1234)
sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
K.set_session(sess)
session_conf.gpu_options.allow_growth = True

input_train = '/home/polina/1tb/ss_prediction/centroid_88_90/data/in/train/'
input_valid = '/home/polina/1tb/ss_prediction/centroid_88_90/data/in/valid/'
output_train = '/home/polina/1tb/ss_prediction/centroid_88_90/data/out/train/'
output_valid = '/home/polina/1tb/ss_prediction/centroid_88_90/data/out/valid/'


bs = 8
epochs = 500
input_shape = (None, None, 1)

def calc_data_size(in_path):
    d = dict()
    files = glob.glob(in_path + '*.png')
    for file in files:
        img = Image.open(file)
        size = img.size[0]
        if size in d.keys():
            d[size] += 1
        else:
            d[size] = 1
        img.close()
    s = 0
    for k in sorted(d.keys()):
        n = d[k]
        while n % bs != 0:
            n += 1
        s += n
    return s

def generate_batches(in_path):
  while True:
    files = glob.glob(in_path + '*.png')
    rn.shuffle(files)
    lens = dict()

#group files by image size   
    for f in files:
        img = np.array(Image.open(f))
        n = len(img)
        if not n in lens.keys():
            lens[n] = []
            lens[n].append(f)
        else:
            lens[n].append(f)
#generate arrays from images of the same size
    for n in lens.keys():
      k = 0
      while len(lens[n]) % bs != 0:
          lens[n].append(lens[n][k])
          k += 1
      X, Y = [], []
      for image_path in lens[n]:
        x = imread(image_path , as_gray=True)
        y = imread(image_path.replace('/in/', '/out/') , as_gray=True)
        X.append(x)
        Y.append(y)
#split them into batches
      for i in range(int(len(X) / bs)):
        shape = (bs, n, n, 1)
        sliceX = np.array(X[i * bs:(i + 1) * bs]).reshape(shape)
        sliceY = np.array(Y[i * bs:(i + 1) * bs]).reshape(shape)
        yield(sliceX, sliceY)

  
def res_unit(inputs, filters, kernel, activ): 
    x = inputs
    x = BatchNormalization()(x)
    for f in filters:
        x = Activation(activ)(x)
        x = Conv2D(f, kernel_size=kernel,
                      padding='same')(x)
    return x


def res_network(units_num, filters, kernel=3, activ = 'relu'):
    inputs = Input(shape=input_shape)
    x = res_unit(inputs, filters, kernel, activ)
    x = add([x, inputs])
    x = Activation(activ)(x)
    for i in range(units_num - 1):
        y = res_unit(x, filters, kernel, activ)
        x = add([x, y])
        x = Activation(activ)(x)
    outputs = x
    
    model = Model(inputs=inputs, outputs=outputs)
    return model

model = res_network(10, [12,10,8,6,1])

coeff = 7
 
def weighted_loss(y_true, y_pred):
    y_true1, y_pred1 = y_true / 255, y_pred / 255
    dif = (1 - K.abs(y_true1 - y_pred1))
    w = (coeff - 1) *y_true1 + 1 
    true_score = K.sum(w, axis = [1,2,3])
    mult = multiply([dif,w])
    pred_score = K.sum(mult, axis = [1,2,3])
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

def f_mera_loss(y_true, y_pred):
    y_true1, y_pred1 = K.minimum(y_true / 255, 1), K.minimum(y_pred / 255, 1)
    fb = K.cast(K.equal(y_true1, 1),"float32") * K.cast(K.less_equal(y_pred1, 0.25),"float32")
    fw = K.cast(K.equal(y_true1, 0),"float32") * K.cast(K.greater(y_pred1, 0.25),"float32")
    tb = K.cast(K.equal(y_true1, 0),"float32") * K.cast(K.less_equal(y_pred1, 0.25),"float32")
    tw = K.cast(K.equal(y_true1, 1),"float32") * K.cast(K.greater(y_pred1, 0.25),"float32")
    fb = K.sum(fb * (y_true1 - y_pred1), axis = [1,2,3])
    fw = K.sum(fw * (y_pred1 - y_true1), axis = [1,2,3])
    tb = K.sum(tb * (1 - y_pred1 + y_true1), axis = [1,2,3])
    tw = K.sum(tw * (1 - y_true1 + y_pred1), axis = [1,2,3])
    prec = tw / (tw + fw + 0.0001)
    rec = tw / (tw + fb + 0.0001)
    f_mera = 2 * prec * rec / (prec + rec + 0.0001)
    return K.mean(1 - f_mera)
 
l1, l2 = 0.5, 0.5

def comb_loss(y_true, y_pred):
    return l1 * weighted_loss(y_true, y_pred) + l2 * f_mera_loss(y_true, y_pred)

model.compile(optimizer=Adagrad(lr=0.08), 
              loss=weighted_loss,
              metrics=[f_mera])

csv_logger = CSVLogger('/home/polina/1tb/ss_prediction/centroid_88_90/models_steps/step1/model1/training.log')


train_size = calc_data_size(input_train)
valid_size = calc_data_size(input_valid)
train_gen = generate_batches(input_train)
valid_gen = generate_batches(input_valid)

model.fit_generator(train_gen,
                    validation_data=valid_gen,
                    steps_per_epoch=int(train_size/bs) - 1,
                    validation_steps=int(valid_size/ bs) - 1,
                    epochs=epochs,
                    verbose=2,
                    shuffle=True,
                    callbacks=[csv_logger])

model.save_weights('/home/polina/1tb/ss_prediction/centroid_88_90/models_steps/step1/model1/weights.h5')

