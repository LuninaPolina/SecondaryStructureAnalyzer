from keras import regularizers
from keras.models import Model
from keras.layers import Dropout, Conv2D, Input, multiply, Activation, BatchNormalization, add
from keras import backend as K
from keras.engine.topology import Layer
from keras.callbacks import CSVLogger, ModelCheckpoint, EarlyStopping
from keras.optimizers import Adagrad
import tensorflow as tf
import numpy as np
import random as rn
from skimage.io import imread
import os
import math
import glob
from PIL import Image


os.environ['PYTHONHASHSEED'] = '0'
np.random.seed(42)
rn.seed(12345)
session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
tf.set_random_seed(1234)
sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
K.set_session(sess)


src_dir = '../../'
model_dir = src_dir + 'models/model1/'
input_dir = src_dir + 'data/in/'
output_dir = src_dir + 'data/out/'

train_bs = 4
epochs = 200
input_shape = (None, None, 1)
len_files = 801


#Data processing functions

def binarize_output(img, coeff=0.25):
    size = len(img)
    for i in range(size):
        for j in range(size):
            if i != j:
                if img[i][j] > 255 * coeff:
                    img[i][j] = 255
                else:
                    img[i][j] = 0
    return img


def compare_images(img_true, img_pred):
    tw, tb, fw, fb = 0, 0, 0, 0
    size = len(img_true)
    for i in range(size):
        for j in range(size):
            if img_true[i][j] != img_pred[i][j] and i != j:
                if int(img_pred[i][j]) == 0:
                    fb += 1
                if int(img_pred[i][j]) == 255:
                    fw += 1
            elif img_true[i][j] == img_pred[i][j] and i != j:
                if int(img_true[i][j]) == 255:
                    tw += 1
                if int(img_true[i][j]) == 0:
                    tb += 1
    prec = tw / (tw + fw + 0.00001)
    rec = tw / (tw + fb + 0.00001)
    f1 = 2 * (prec * rec) / (prec + rec + 0.00001)
    precs.append(prec)
    recs.append(rec)
    f1s.append(f1)
                

def estimate(f_true, f_pred):
    with Image.open(f_true) as img_true, Image.open(f_pred) as img_pred:
        img_true = np.array(img_true)
        img_pred = binarize_output(np.array(img_pred))
        compare_images(img_true, img_pred)


def load_image(f):
    img_arr = []
    img = imread(f , as_gray=True)
    n = img.shape[0]
    img_arr.append(img)
    return np.array(img_arr).reshape(1,n,n,1)


def split_files(train_percent):
    global train_files, test_files
    files = glob.glob(input_dir + '*.png')
    rn.seed()
    rn.shuffle(files)
    cnt = 0
    for f in files:
        if cnt < len_files * train_percent:
            train_files.append(f)
        else:
            test_files.append(f)
        cnt += 1


def calc_data_size(in_path, dataset, bs, mirror=False, repeat=1):
    d = dict()
    files = glob.glob(in_path + '*.png')
    for file in files:
        if file in dataset:
            img = Image.open(file)
            size = img.size[0]
            cnt = 1
            if mirror:
                cnt *=  2
            cnt *= repeat
            if size in d.keys():
                d[size] += cnt
            else:
                d[size] = cnt
            img.close()
    s = 0
    for k in sorted(d.keys()):
        n = d[k]
        while n % bs != 0:
            n += 1
        s += n
    return s


def mirror_diag(f):
    img = np.array(Image.open(f))
    n = len(img)
    img2 = np.array(Image.new('L', (n, n), (0)))
    for i in range(n):
        for j in range(n):
            img2[i][j] = img[n - j - 1][n - i - 1]
    f2 = f.split('.png')[0] + '_2.png'
    Image.fromarray(img2).save(f2)


def generate_batches(in_path, dataset, bs, mirror=False, repeat=1):
  while True:
      files = glob.glob(in_path + '*.png')
      if mirror:
          for f in files:
              if f in dataset:
                  mirror_diag(f)
                  mirror_diag(f.replace(input_dir, output_dir))
      lens = dict()
      files = []
      for i in range(repeat):
          files += glob.glob(in_path + '*.png')
      rn.seed()
      rn.shuffle(files)  
      for f in files:
          if f in dataset or f.replace('_2.png', '.png') in dataset:
              img = np.array(Image.open(f))
              n = len(img)
              if not n in lens.keys():
                  lens[n] = []
                  lens[n].append(f)
              else:
                  lens[n].append(f)
      for n in lens.keys():
          k = 0
          while len(lens[n]) % bs != 0:
              lens[n].append(lens[n][k])
              k += 1
          X, Y = [], []
          for image_path in lens[n]:
              x = imread(image_path , as_gray=True)
              y = imread(image_path.replace(input_dir, output_dir) , as_gray=True)
              X.append(x)
              Y.append(y)
          for i in range(int(len(X) / bs)):
              shape = (bs, n, n, 1)
              sliceX = np.array(X[i * bs:(i + 1) * bs]).reshape(shape)
              sliceY = np.array(Y[i * bs:(i + 1) * bs]).reshape(shape)
              yield(sliceX, sliceY)


#Model definition functions

class WeightedSum(Layer):

    def __init__(self):
        super(WeightedSum, self).__init__()
        w_init = tf.keras.initializers.Ones()
        self.w1 = tf.Variable(initial_value=w_init(shape=(), dtype='float32'), trainable=True)
        self.w2 = tf.Variable(initial_value=w_init(shape=(), dtype='float32'), trainable=True)  
        self.w3 = tf.Variable(initial_value=w_init(shape=(), dtype='float32'), trainable=True)   
        self.w4 = tf.Variable(initial_value=w_init(shape=(), dtype='float32'), trainable=True)       

    def call(self, input):
        input1, input2, input3, input4 = input
        return (tf.multiply(input1,self.w1) + tf.multiply(input2, self.w2) + tf.multiply(input3, self.w3) + tf.multiply(input4, self.w4)) / 4


def res_unit(inputs, filters, kernels, activ, bn=False, dr=0): 
    x = inputs
    if bn: 
    	x = BatchNormalization()(x)
    for i in range(len(filters)):
        x = Activation(activ)(x)
        x = Conv2D(filters[i], kernel_size=kernels[i], activity_regularizer=regularizers.l2(1e-10),padding='same')(x)
        if dr > 0:
        	x = Dropout(0.1)(x)
    return x


def res_network(inputs, units_num, filters, kernels, activ='relu', bn=False, dr=0):
    x = res_unit(inputs, filters, kernels, activ, bn, dr)
    x = add([x, inputs])
    x = Activation(activ)(x)
    for i in range(units_num - 1):
        y = res_unit(x, filters, kernels, activ, bn, dr)
        x = add([x, y])
        x = Activation(activ)(x)
    outputs = x  

    model = Model(inputs=inputs, outputs=outputs)
    return model


def parallel_res_network(blocks_num, units_num, filters, kernels, activ='relu', bn=False, dr=0):
    inputs = Input(shape=input_shape)
    all_outputs = []
    for i in range(blocks_num):
        model = res_network(inputs, units_num, filters, kernels, activ, bn, dr)
        all_outputs.append(model.output)
    
    x = WeightedSum()(all_outputs)

    y = res_unit(x, filters, kernels, activ, bn, dr)
    x = add([x, y])
    x = Activation(activ)(x)
    outputs = x
    model = Model(inputs=inputs, outputs=outputs)
    return model


def precision(y_true, y_pred):
    y_true1, y_pred1 = K.minimum(y_true / 255, 1), K.minimum(y_pred / 255, 1)
    fb = K.cast(K.equal(y_true1, 1),'float32') * K.cast(K.less_equal(y_pred1, 0.25),'float32')
    fw = K.cast(K.equal(y_true1, 0),'float32') * K.cast(K.greater(y_pred1, 0.25),'float32')
    tb = K.cast(K.equal(y_true1, 0),'float32') * K.cast(K.less_equal(y_pred1, 0.25),'float32')
    tw = K.cast(K.equal(y_true1, 1),'float32') * K.cast(K.greater(y_pred1, 0.25),'float32')
    fb = K.sum(fb, axis = [1,2,3])
    fw = K.sum(fw, axis = [1,2,3])
    tb = K.sum(tb, axis = [1,2,3])
    tw = K.sum(tw, axis = [1,2,3])
    prec = tw / (tw + fw + K.epsilon())
    return K.mean(prec)


def recall(y_true, y_pred):
    y_true1, y_pred1 = K.minimum(y_true / 255, 1), K.minimum(y_pred / 255, 1)
    fb = K.cast(K.equal(y_true1, 1),'float32') * K.cast(K.less_equal(y_pred1, 0.25),'float32')
    fw = K.cast(K.equal(y_true1, 0),'float32') * K.cast(K.greater(y_pred1, 0.25),'float32')
    tb = K.cast(K.equal(y_true1, 0),'float32') * K.cast(K.less_equal(y_pred1, 0.25),'float32')
    tw = K.cast(K.equal(y_true1, 1),'float32') * K.cast(K.greater(y_pred1, 0.25),'float32')
    fb = K.sum(fb, axis = [1,2,3])
    fw = K.sum(fw, axis = [1,2,3])
    tb = K.sum(tb, axis = [1,2,3])
    tw = K.sum(tw, axis = [1,2,3])
    rec = tw / (tw + fb + K.epsilon())
    return K.mean(rec)
    

def f1_metrics(y_true, y_pred):
    y_true1, y_pred1 = K.minimum(y_true / 255, 1), K.minimum(y_pred / 255, 1)
    y_true1 = process_diag(y_true1)[0]
    y_pred1 = process_diag(y_pred1)[0]
    fb = K.cast(K.equal(y_true1, 1),'float32') * K.cast(K.less_equal(y_pred1, 0.25),'float32')
    fw = K.cast(K.equal(y_true1, 0),'float32') * K.cast(K.greater(y_pred1, 0.25),'float32')
    tb = K.cast(K.equal(y_true1, 0),'float32') * K.cast(K.less_equal(y_pred1, 0.25),'float32')
    tw = K.cast(K.equal(y_true1, 1),'float32') * K.cast(K.greater(y_pred1, 0.25),'float32')
    fb = K.sum(fb, axis = [1,2,3]) 
    fw = K.sum(fw, axis = [1,2,3])
    tb = K.sum(tb, axis = [1,2,3])
    tw = K.sum(tw, axis = [1,2,3])
    prec = tw / (tw + fw + K.epsilon())
    rec = tw / (tw + fb + K.epsilon())
    f1 = 2 * prec * rec / (prec + rec + K.epsilon())
    return K.mean(f1)


def process_diag(tensor):
    tensor = K.cast(tf.squeeze(tensor, axis=[-1]), 'float32')
    diag = tf.expand_dims(tf.matrix_diag(tf.matrix_diag_part(tensor)), -1)
    zeros = tf.zeros(tf.shape(tensor)[0:-1], dtype=tf.float32)
    tensor = tf.expand_dims(tf.linalg.set_diag(tensor, zeros, name=None), -1)
    return tensor, diag 


def f1_loss(y_true, y_pred):
    y_true1, y_pred1 = K.minimum(y_true / 255, 1), K.minimum(y_pred / 255, 1)
    y_true1, y_true1_diag = process_diag(y_true1)
    y_pred1, y_pred1_diag = process_diag(y_pred1)
    tw = K.sum(K.cast(y_true1 * y_pred1, 'float32'), axis=[1, 2, 3])
    tb = K.sum(K.cast((1 - y_true1) * (1 - y_pred1), 'float32'), axis=[1, 2, 3])
    fw = K.sum(K.cast((1 - y_true1) * y_pred1, 'float32'), axis=[1, 2, 3])
    fb = K.sum(K.cast(y_true1 * (1 - y_pred1), 'float32'), axis=[1, 2, 3])
    prec = tw / (tw + fw + K.epsilon())
    rec = tw / (tw + fb + K.epsilon())
    k1 = 1 -  K.abs(prec - rec)
    k2 = 1 -  K.abs(K.mean(prec) - K.mean(rec))
    k3 = 1 - K.sum(K.abs(y_true1_diag - y_pred1_diag), axis=[1,2,3]) / K.cast(tf.shape(y_true1_diag)[1], 'float32')
    f1 = k1 * k2 * k3 * 2 * prec * rec / (prec + rec + K.epsilon()) 
    return 1 - K.mean(f1)


#Build model, generate data, launch training and testing

start, stop, step = 10, 100, 10

for percent in range(start, stop, step):
    prec, rec, f1 = 0, 0, 0
    files = glob.glob(input_dir + '*.png')
    for f in files:
        if '_2' in f:
            os.remove(f)
            os.remove(f.replace(input_dir, output_dir))

    model = parallel_res_network(blocks_num=4, units_num=5, filters=[12, 10, 8, 6, 1], kernels=[13, 11, 9, 7, 5], activ='relu', bn=False, dr=0.1)
    model.compile(optimizer=Adagrad(lr=0.005), loss=f1_loss, metrics=[precision, recall, f1_metrics])

    cl = CSVLogger(model_dir + 'training_' + str(percent) + '.log')
    es = EarlyStopping(monitor='val_f1_metrics', mode='max', verbose=1, patience=30)
    mc = ModelCheckpoint(model_dir + 'weights_' + str(percent) + '.h5', save_best_only=True, monitor='val_f1_metrics', mode='max')

    train_files, test_files = [], []
    split_files(0.01 * percent)
    open(model_dir + 'datasets.txt', 'a').write(str(percent) + '%\ntrain:\n' + '\n'.join(train_files) + '\ntest:\n' + '\n'.join(test_files) + '\n')

    train_size = calc_data_size(input_dir, train_files, train_bs, mirror=True, repeat=2) 
    test_size = calc_data_size(input_dir, test_files, 1)
    print(train_size, test_size)
    train_gen = generate_batches(input_dir, train_files, train_bs, mirror=True, repeat=2)
    test_gen = generate_batches(input_dir, test_files, 1)

    model.fit_generator(train_gen,
                        validation_data=test_gen,
                        steps_per_epoch=int(train_size/train_bs),
                        validation_steps=test_size,
                        epochs=epochs,
                        verbose=2,
                        shuffle=True,
                        callbacks=[cl, es, mc])

    print('Training done for', percent, '% training data')

    model = parallel_res_network(blocks_num=4, units_num=5, filters=[12, 10, 8, 6, 1], kernels=[13, 11, 9, 7, 5], activ='relu', bn=False, dr=0.1)
    model.load_weights(model_dir + 'weights_'  + str(percent) + '.h5')
    pred_dir = model_dir + 'predicted/'
    if not os.path.exists(pred_dir):
        os.mkdir(pred_dir)
    files = glob.glob(input_dir + '*.png')
    for f in files:
        if f in test_files:
            img = load_image(f)
            res = model.predict(img)
            k = len(img[0])
            pixels = np.array(Image.new('L', (k, k), (255)))
            for i in range(k):
                for j in range(k):
                    pixels[i][j] = min(res[0][i][j][0], 255)
            Image.fromarray(pixels, 'L').save(f.replace(input_dir, pred_dir))
    print('Prediction done for', 100 - percent, '% test data')
    precs, recs, f1s = [], [], []
    files_true = sorted(glob.glob(output_dir + '*.png'))
    files_pred = sorted(glob.glob(pred_dir + '*.png'))
    for f in files_pred:
        estimate(f.replace(pred_dir, output_dir), f)
    prec = sum(precs) / len(precs)
    rec = sum(recs) / len(recs)
    f1 = sum(f1s) / len(f1s)
    print('Precision =', prec, '\nRecall =', rec, '\nF1 =', f1)
    os.system('rm -r ' + pred_dir)
    files = glob.glob(input_dir + '*.png')
    for f in files:
        if '_2' in f:
            os.remove(f)
            os.remove(f.replace(input_dir, output_dir))
    K.clear_session()
