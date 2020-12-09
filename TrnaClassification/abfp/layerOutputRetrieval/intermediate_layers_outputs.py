from keras.models import Model, Sequential
from keras.layers import Activation, Dropout, BatchNormalization, Dense
from keras import backend as K

from PIL import Image, ImageOps
import pandas as pd
import csv
from tqdm import tqdm
import os
import numpy as np
import tensorflow as tf
import random as rn


# Для одинакового результата при разных запусках.
os.environ['PYTHONHASHSEED'] = '0'
np.random.seed(42)
rn.seed(12345)
session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
tf.set_random_seed(1234)
sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
K.set_session(sess)

# specify data paths here
data_vec = 'for_report.csv'
db_file_vec = 'ref_db.csv'
weights_vec = 'weights_vec.h5'
out_dir_vec = 'intermediate_output_vec'

data_img = 'test.csv'
db_file_img = 'ref_db.csv'
weights_img = 'weights_img.h5'
out_dir_img = 'intermediate_output_img'


def generate_arrays(path, db):
    db = pd.read_csv(db)
    f = open(path)
    r = csv.reader(f)
    batch_x = []
    batch_y = []
    for ln in r:
        if len(ln) > 0:
            for i in range(1,len(ln)):
                if ln[i] == "A": ln[i] = "2"
                elif ln[i] == "C": ln[i] = "3"
                elif ln[i] == "G": ln[i] = "5"
                elif ln[i] == "T": ln[i] = "7"
                else: ln[i] = "0"
            X = np.array(ln[1:len(ln)],dtype=np.uint32)
            y = [1, 0, 0, 0]
            if db.loc[db['id'] == int(ln[0][1:])].values[0][2] == 'b':
                y = [0, 1, 0, 0]
            if db.loc[db['id'] == int(ln[0][1:])].values[0][2] == 'f':
                y = [0, 0, 1, 0]
            if db.loc[db['id'] == int(ln[0][1:])].values[0][2] == 'p':
                y = [0, 0, 0, 1]
            batch_x.append(np.array(X))
            batch_y.append(y)
    return np.array(batch_x), np.array(batch_y)


def vec2img(vec, length, save_path):
    bits = []
    for el in vec[1:]:
        num = int(el)
        binary = '{:032b}'.format(abs(num))[::-1]
        for b in binary: bits.append(int(b))
    bits = bits[:24200] + [0 for i in range(110)]
    mtx = []
    fr, to = 0, 220
    for i in range(length):
        mtx.append([0 for k in range(i)] + bits[fr:to])
        fr = to
        to = to + 220 - i - 1
    pixels = []
    for row in mtx:
        for b in row:
            if b == 1:
                pixels.append((0, 0, 0))
            else:
                pixels.append((255, 255, 255))
    img = Image.new('RGB', (length, length), (255,255,255))
    img.putdata(pixels)
    img = ImageOps.mirror(img).rotate(90)
    img.save(save_path)


def load_img_model(weights):
    input_length = 220

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

    # We name the layer that interests us here to be able to retrieve it later
    model.add(Dense(30420, name='target_layer'))
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
    return model


def save_img_model_results(data, weights, db_file, out_dir):
    """
    Retrieves intermediate layer results and saves them into out_dir
    """
    channels_first_flag = K.image_data_format() == 'channels_first'  # find out data format
    data = generate_arrays(data, db_file)
    model = load_img_model(weights)
    # define a model with intermediate layers
    intermediate_layer_model = Model(inputs=model.input, outputs=model.get_layer('target_layer').output)
    # get intermediate output
    intermediate_output = intermediate_layer_model.predict(data[0])
    # reshape and save results
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    for i in tqdm(range(intermediate_output.shape[0])):
        if channels_first_flag:
            result = np.reshape(intermediate_output[i], (5, 78, 78))
        else:
            result = np.reshape(intermediate_output[i], (78, 78, 5))
        path = os.path.join(out_dir, f'arr_{i}.npy')
        np.save(path, result)


def load_img_model_results(folder):
    file_names = os.listdir(folder)
    file_paths = list(map(lambda x: os.path.join(folder, x), file_names))
    results_list = []
    for path in file_paths:
        arr = np.load(path)
        results_list.append(arr)
    return results_list


def load_vec_model(weights):
    arr_length = 3028
    input_length = 220

    model = Sequential()

    model.add(Dropout(0.3, input_shape=(arr_length,)))

    model.add(Dense(8194, trainable=False))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(Dropout(0.9))

    model.add(Dense(2048, trainable=False))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(Dropout(0.9))

    model.add(Dense(1024, trainable=False))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(Dropout(0.9))


    model.add(Dense(512, trainable=False))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(Dropout(0.75))

    model.add(Dense(64, trainable=False))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(Dropout(0.5))

    model.add(Dense(4, trainable=False))
    model.add(Activation('softmax'))

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

    # We name the layer that interests us here to be able to retrieve it later
    model2.add(Dense(3028))
    model2.add(BatchNormalization())
    model2.add(Activation('relu', name='target_layer'))


    model2.add(model)

    model2.load_weights(weights)
    return model2


def visualize_vec_model(data, weights, db_file, out_dir):
    """
    Retrieves intermediate layer results and saves them into out_dir
    """
    data = generate_arrays(data, db_file)
    model_vec = load_vec_model(weights)
    # define a model with intermediate layers
    intermediate_layer_model = Model(inputs=model_vec.input, outputs=model_vec.get_layer('target_layer').output)
    # get intermediate output
    intermediate_output = intermediate_layer_model.predict(data[0])

    # visualise and save results
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    for i in tqdm(range(intermediate_output.shape[0])):
        save_path = f'out_{i}.bmp'
        save_path = os.path.join(out_dir, save_path)
        vec2img(intermediate_output[i], 220, save_path)


visualize_vec_model(data_vec, weights_vec, db_file_vec, out_dir_vec)
save_img_model_results(data_img, weights_img, db_file_img, out_dir_img)
img_res = load_img_model_results(out_dir_img)
print(img_res[0].shape)
