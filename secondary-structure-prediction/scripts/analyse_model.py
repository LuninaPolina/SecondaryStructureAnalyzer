'''Functions that analyse neural network intermediate layers outputs'''

from keras import regularizers
from keras.models import Model
from keras.layers import Dropout, Conv2D, Input, multiply, Activation, BatchNormalization, add
from keras import backend as K
from keras.engine.topology import Layer
from keras.callbacks import CSVLogger, ModelCheckpoint, EarlyStopping
from keras.optimizers import Adagrad
import tensorflow as tf
import numpy as np
from skimage.io import imread
import os
import glob
from PIL import Image
from shutil import copyfile
import imageio


#Model definition function

input_shape = (None, None, 1)


def load_image(f):
    img_arr = []
    img = imread(f , as_gray=True)
    n = img.shape[0]
    img_arr.append(img)
    return np.array(img_arr).reshape(1,n,n,1)


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


def mkdir(dir):
    if not os.path.exists(dir):
        os.mkdir(dir)


#Analytics functions

def load_model(weights_file):
    model = parallel_res_network(blocks_num=4, units_num=5, filters=[12, 10, 8, 6, 1], kernels=[13, 11, 9, 7, 5], activ='relu', bn=False, dr=0.1)
    model.load_weights(weights_file)
    return model


def get_last4_layers(model):
    last4 = ['', '', '', '']
    for i in range(1, len(model.layers)):
        if 'weighted_sum' in model.layers[i].name:
            return last4[4:]
        else:
            last4[(i - 1) % 4] = model.layers[i].name
        

def resnet_heads_out(in_file, weights_file, out_dir, id):
    inp = load_image(in_file)
    model = load_model(weights_file)
    out = model.predict(inp)
    k = len(out[0])
    pixels = np.array(Image.new('L', (k, k), (255)))
    for i in range(k):
        for j in range(k):
            pixels[i][j] = min(out[0][i][j][0], 255)
    Image.fromarray(pixels, 'L').save(out_dir + id + '_pred.png')
    target_layers = get_target_layers(model)
    cnt = 1
    for tl in target_layers:
        intermediate_model = Model(inputs=model.input, outputs=model.get_last4_layers(tl).output)
        intermediate_out = intermediate_model.predict(inp)
        k = len(intermediate_out[0])
        pixels = np.array(Image.new('L', (k, k), (255)))
        for i in range(k):
            for j in range(k):
                pixels[i][j] = min(intermediate_out[0][i][j][0], 255)
        Image.fromarray(pixels, 'L').save(out_dir + str(id) + '_resnet_' + str(cnt) + '.png')
        cnt += 1
    
    
def analyze(data_dir, out_dir, weights_fil, id):
    in_file = data_dir + 'in/' + id + '.png'
    out_file = data_dir + 'out/' + id + '.png'
    copyfile(in_file, out_dir + id + '_in.png')
    copyfile(out_file, out_dir + id + '_out.png')
    resnet_heads_out(in_file, weights_file,  out_dir, id)


def get_all_intermediate(in_file, weights_file, out_dir, id, predict=True):
    mkdir(out_dir + id + '/')
    inp = load_image(in_file)
    model = load_model(weights_file)
    out = model.predict(inp)
    cnt = 0
    cnt0 = 0
    seq_order = []
    for l in model.layers:
        if cnt == 0:
            idx = '0_0'
        else:
            if 'weighted_sum' in l.name:
                cnt0 = 1
            if cnt0 > 0:
                idx = '0_' + str(cnt0)
                cnt0 += 1
            else:
                idx = str((cnt - 1) % 4 + 1) + '_' + str(int((cnt - 1) / 4) + 1)
        if predict:
            if cnt == 0:
                out = model.predict(inp)
                k = len(out[0])
                pixels = np.array(Image.new('L', (k, k), (255)))
                for i in range(k):
                    for j in range(k):
                        pixels[i][j] = min(out[0][i][j][0], 255)
                Image.fromarray(pixels, 'L').save(out_dir + id + '/' + idx + '.png')
            else:
                intermediate_model = Model(inputs=model.input, outputs=model.get_layer(l.name).output)
                intermediate_out = intermediate_model.predict(inp)
                k = len(intermediate_out[0])
                pixels = np.array(Image.new('L', (k, k), (255)))
                for i in range(k):
                    for j in range(k):
                        pixels[i][j] = min(intermediate_out[0][i][j][0], 255)
                Image.fromarray(pixels, 'L').save(out_dir + id + '/' + idx + '.png')
        seq_order.append(idx)
        cnt += 1
    return seq_order


def animate(id, data_dir, weights_file, out_dir):
    in_file = data_dir + 'in/' + id + '.png'
    seq_order = get_all_intermediate(in_file, weights_file, out_dir, id, True)
    images = []
    cnt = 0
    for idx in seq_order:
        f = out_dir + str(id) + '/' + idx + '.png'
        img = Image.open(f)
        n = img.size[0]
        if '0_' in f:
            frame = Image.new('L', (9 * n, 7 * n), (255)) 
            if '0_0' in f:
                frame.paste(img, (4 * n, n))
                for i in range(5):
                    frame.save(out_dir + str(id) + '/frame_' + ''.join([str(cnt) for j in range(i + 1)]) + '.png')
                    images.append(imageio.imread(out_dir + str(id) + '/frame_' + ''.join([str(cnt) for j in range(i + 1)]) + '.png'))
            elif '0_18' in f:
                frame.paste(img, (4 * n, 5 * n))
                for i in range(5):
                    frame.save(out_dir + str(id) + '/frame_' + ''.join([str(cnt) for j in range(i + 1)]) + '.png')
                    images.append(imageio.imread(out_dir + str(id) + '/frame_' + ''.join([str(cnt) for j in range(i + 1)]) + '.png'))
            else:
                frame.paste(img, (4 * n, 5 * n))
            frame.save(out_dir + str(id) + '/frame_' + str(cnt) + '.png')
            images.append(imageio.imread(out_dir + str(id) + '/frame_' + str(cnt) + '.png'))
        if '1_' in f:
            frame = Image.new('L', (9 * n, 7 * n), (255)) 
            frame.paste(img, (n, 3 * n))
        if '2_' in f:
            frame.paste(img, (3 * n, 3 * n))
        if '3_' in f:
            frame.paste(img, (5 * n, 3 * n))
        if '4_' in f:
            frame.paste(img, (7 * n, 3 * n))
            frame.save(out_dir + str(id) + '/frame_' + str(cnt) + '.png')
            images.append(imageio.imread(out_dir + str(id) + '/frame_' + str(cnt) + '.png'))
        cnt += 1
    imageio.mimsave(out_dir + str(id) + '/video.mp4', images, 'mp4', fps=4)

