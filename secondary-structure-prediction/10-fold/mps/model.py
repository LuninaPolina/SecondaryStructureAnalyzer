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


#specify paths here
src_dir = ''
model_dir = src_dir + 'main/'
input_dir = src_dir + 'data/in/'
output_dir = src_dir + 'data/out/'

train_bs = 4
epochs = 200
input_shape = (None, None, 1)


#Data processing functions

def binarize_output(img, coeff=0.25): #set each gray pixel of network prediction to black/white according to threshold coeff
    size = len(img)
    for i in range(size):
        for j in range(size):
            if i != j:
                if img[i][j] > 255 * coeff:
                    img[i][j] = 255
                else:
                    img[i][j] = 0
    return img

def get_multiplets(i0, j0, img):
    multiplets = []
    size = len(img)
    for i in range(size):
        if img[i0, i] == 255 and (i0, i) != (i0, j0) and i0 <= i:
            multiplets.append((i0, i))
        if img[j0, i] == 255 and (j0, i) != (i0, j0) and j0 <= i:
            multiplets.append((j0, i))
        if img[i, i0] == 255 and (i, i0) != (i0, j0) and i <= i0:
            multiplets.append((i, i0))
        if img[i, j0] == 255 and (i, j0) != (i0, j0) and i <= j0:
            multiplets.append((i, j0))
    return list(set(multiplets))

def get_mps_pairs(img):
    size = len(img)
    mps_pairs = []
    for i in range(size):
        for j in range(i + 1, size):
            if img[i][j] == 255:
                mps = get_multiplets(i, j, img)
                if len(mps) > 0:
                    for m in mps:
                        if not ((i, j), m) in mps_pairs and not (m, (i, j)) in mps_pairs:
                            mps_pairs.append(((min(i, j), max(i, j)), (min(m[0], m[1]), max(m[0], m[1]))))

    return mps_pairs

def estimate_mps(in_dir, out_dir):
    files = glob.glob(in_dir + '*.png')
    cnt_mps, cnt_true_mps, cnt_cont, cnt_true_cont = 0, 0, 0, 0
    for f_pred in files:
        if f_pred.split('/')[-1].split('.')[0] in val_ids.split(' '):
            f_true = f_pred.replace(in_dir, out_dir)
            img_true = np.array(Image.open(f_true))
            img_pred = np.array(Image.open(f_pred))
            img_pred = binarize_output(img_pred, 0.2)
            mps_true = get_mps_pairs(img_true)
            mps_pred = get_mps_pairs(img_pred)
            cnt_mps += len(mps_true)
            for (m1, m2) in mps_true:
                if (m1, m2) in mps_pred or (m2, m1) in mps_pred:
                    cnt_true_mps += 1
            cont_true = list(set([x[0] for x in mps_true] + [x[1] for x in mps_true]))
            cont_pred = list(set([x[0] for x in mps_pred] + [x[1] for x in mps_pred]))
            cnt_cont += len(cont_true)
            for c in cont_true:
                if c in cont_pred:
                    cnt_true_cont += 1

    print('predicted mps', cnt_true_mps, 'from', cnt_mps)
    print('predicted mps contacts', cnt_true_cont, 'from', cnt_cont)

def compare_images(img_true, img_pred): #calculate precision, recall and f1 for reference and prediction pictures
    tw, tb, fw, fb = 0, 0, 0, 0
    size = len(img_true)
    for i in range(size):
        for j in range(i + 1, size):
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
                

def estimate(f_true, f_pred): #process reference and prediction files
    with Image.open(f_true) as img_true, Image.open(f_pred) as img_pred:
        img_true = np.array(img_true)
        img_pred = binarize_output(np.array(img_pred))
        compare_images(img_true, img_pred)


def load_image(f): #load one image in format accepted by network (required for tests)
    img_arr = []
    img = imread(f , as_gray=True)
    n = img.shape[0]
    img_arr.append(img)
    return np.array(img_arr).reshape(1,n,n,1)


def split_files_kfold(k):
    folds = []
    files = glob.glob(input_dir + '*.png')
    rn.seed()
    rn.shuffle(files)
    for i in range(k):
        start = int(i * len(files) / k)
        stop =  int((i + 1) * len(files) / k)
        folds.append(files[start:stop])
    return folds

def mirror_diag(f): #mirror image (data augmentation technique)
    img = np.array(Image.open(f))
    n = len(img)
    img2 = np.array(Image.new('L', (n, n), (0)))
    for i in range(n):
        for j in range(n):
            img2[i][j] = img[n - j - 1][n - i - 1]
    f2 = f.split('.png')[0] + '_mirror.png'
    Image.fromarray(img2).save(f2)


def generate_batches(in_path, dataset, bs, mirror=False, repeat=1): #batch generator that combines images with same size
  while True:                                                       #current possible augmentations: mirror image, repeat image n times
      files = glob.glob(in_path + '*.png')
      if mirror: #mirror images if required
          for f in files:
              if f in dataset:
                  mirror_diag(f)
                  mirror_diag(f.replace(input_dir, output_dir))
      lens = dict()
      files = []
      for i in range(repeat): #repeat images if required
          files += glob.glob(in_path + '*.png')
      rn.seed()
      rn.shuffle(files)  
      for f in files: #sort files by image length
          if f in dataset or f.replace('_mirror.png', '.png') in dataset:
              img = np.array(Image.open(f))
              n = len(img)
              if not n in lens.keys():
                  lens[n] = []
                  lens[n].append(f)
              else:
                  lens[n].append(f)
      for n in lens.keys(): #cyclically add files until all bathes have size equal to argument bs 
          k = 0
          while len(lens[n]) % bs != 0:
              lens[n].append(lens[n][k])
              k += 1
          X, Y = [], []
          for image_path in lens[n]: #yield bathes for train data (X) and reference data (Y)
              x = imread(image_path , as_gray=True)
              y = imread(image_path.replace(input_dir, output_dir) , as_gray=True)
              X.append(x)
              Y.append(y)
          for i in range(int(len(X) / bs)):
              shape = (bs, n, n, 1)
              sliceX = np.array(X[i * bs:(i + 1) * bs]).reshape(shape)
              sliceY = np.array(Y[i * bs:(i + 1) * bs]).reshape(shape)
              yield(sliceX, sliceY)


def calc_data_size(in_path, dataset, bs, mirror=False, repeat=1): #calculate data size, because after repeating files to filling the batches it can increase
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


#Model definition functions

class WeightedSum(Layer): #layer that for inputs i1, i2, i3, i4 returns (w1*i1 + w2*i2 + w3*i3 + w4*i4) / 4, where wi are trainable coeffs
    def __init__(self):   #for another amount of inputs this class should be changed manually!
        super(WeightedSum, self).__init__()
        w_init = tf.keras.initializers.Ones()
        self.w1 = tf.Variable(initial_value=w_init(shape=(), dtype='float32'), trainable=True)
        self.w2 = tf.Variable(initial_value=w_init(shape=(), dtype='float32'), trainable=True)  
        self.w3 = tf.Variable(initial_value=w_init(shape=(), dtype='float32'), trainable=True)   
        self.w4 = tf.Variable(initial_value=w_init(shape=(), dtype='float32'), trainable=True)       

    def call(self, input):
        input1, input2, input3, input4 = input
        return (tf.multiply(input1,self.w1) + tf.multiply(input2, self.w2) + tf.multiply(input3, self.w3) + tf.multiply(input4, self.w4)) / 4


def res_unit(inputs, filters, kernels, activ, bn=False, dr=0): #residual unit definition, classical structure in ML
    x = inputs
    if bn: 
        x = BatchNormalization()(x)
    for i in range(len(filters)):
        x = Activation(activ)(x)
        x = Conv2D(filters[i], kernel_size=kernels[i], activity_regularizer=regularizers.l2(1e-10),padding='same')(x)
        if dr > 0:
            x = Dropout(0.1)(x)
    return x


def res_network(inputs, units_num, filters, kernels, activ='relu', bn=False, dr=0): #residual network definition: repeating res units + skip sonnections(add layers)
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


def parallel_res_network(blocks_num, units_num, filters, kernels, activ='relu', bn=False, dr=0): #model that combines several residual networks
    inputs = Input(shape=input_shape)
    all_outputs = []
    for i in range(blocks_num): #construct several resnets of same shape
        model = res_network(inputs, units_num, filters, kernels, activ, bn, dr)
        all_outputs.append(model.output)
    
    x = WeightedSum()(all_outputs) #combine their outputs by weighted sum

    y = res_unit(x, filters, kernels, activ, bn, dr) #add final res unit to the end of network
    x = add([x, y])
    x = Activation(activ)(x)
    outputs = x
    model = Model(inputs=inputs, outputs=outputs)
    return model


def process_diag(tensor): #set elements below diagonal to zero (because we don't need them) while calculating all metrics and losses
    tensor = K.cast(tf.squeeze(tensor, axis=[-1]), 'float32')
    tensor = tf.linalg.band_part(tensor, num_lower = 0, num_upper = -1, name=None)
    zeros = tf.zeros(tf.shape(tensor)[0:-1], dtype=tf.float32)
    tensor = tf.expand_dims(tf.linalg.set_diag(tensor, zeros, name=None), -1)
    return tensor 


#just classical precision, recall and f1 in tensor form
def precision(y_true, y_pred): 
    y_true1, y_pred1 = K.minimum(y_true / 255, 1), K.minimum(y_pred / 255, 1)
    y_true1, y_pred1 = process_diag(y_true1), process_diag(y_pred1)
    tw = K.sum(K.cast(y_true1 * y_pred1, 'float32'), axis=[1, 2, 3])
    fw = K.sum(K.cast((1 - y_true1) * y_pred1, 'float32'), axis=[1, 2, 3])
    prec = tw / (tw + fw + K.epsilon())
    return K.mean(prec)


def recall(y_true, y_pred):
    y_true1, y_pred1 = K.minimum(y_true / 255, 1), K.minimum(y_pred / 255, 1)
    y_true1, y_pred1 = process_diag(y_true1), process_diag(y_pred1)
    tw = K.sum(K.cast(y_true1 * y_pred1, 'float32'), axis=[1, 2, 3])
    fb = K.sum(K.cast(y_true1 * (1 - y_pred1), 'float32'), axis=[1, 2, 3])
    rec = tw / (tw + fb + K.epsilon())
    return K.mean(rec)


def f1_metrics(y_true, y_pred):
    y_true1, y_pred1 = K.minimum(y_true / 255, 1), K.minimum(y_pred / 255, 1)
    y_true1, y_pred1 = process_diag(y_true1), process_diag(y_pred1)
    tw = K.sum(K.cast(y_true1 * y_pred1, 'float32'), axis=[1, 2, 3])
    fw = K.sum(K.cast((1 - y_true1) * y_pred1, 'float32'), axis=[1, 2, 3])
    fb = K.sum(K.cast(y_true1 * (1 - y_pred1), 'float32'), axis=[1, 2, 3])
    prec = tw / (tw + fw + K.epsilon())
    rec = tw / (tw + fb + K.epsilon())
    f1 = 2 * prec * rec / (prec + rec + K.epsilon())
    return K.mean(f1)


def f1_loss(y_true, y_pred): #loss that is based on minimizing 1 - f1_metrics
    y_true1, y_pred1 = K.minimum(y_true / 255, 1), K.minimum(y_pred / 255, 1)
    y_true1, y_pred1 = process_diag(y_true1), process_diag(y_pred1)
    tw = K.sum(K.cast(y_true1 * y_pred1, 'float32'), axis=[1, 2, 3])
    fw = K.sum(K.cast((1 - y_true1) * y_pred1, 'float32'), axis=[1, 2, 3])
    fb = K.sum(K.cast(y_true1 * (1 - y_pred1), 'float32'), axis=[1, 2, 3])
    prec = tw / (tw + fw + K.epsilon())
    rec =  tw / (tw + fb + K.epsilon())
#two penalties, because we don't want precision ~ 100% and recall ~ 0% (or the oposite) -- the perfect case is when they are almost equal
    k1 = 1 -  K.abs(prec - rec) #penalty for huge difference in each picture
    k2 = 1 -  K.abs(K.mean(prec) - K.mean(rec)) #same, but generally for all dataset 
    prec = 100 * prec
    f1 = k1 * k2 * 2 * prec * rec / (prec + rec + K.epsilon()) 
    return 1 - K.mean(f1)

def f1_mps_loss(y_true, y_pred): #f1 loss that can consider the quality of multiplets prediction 
    y_true1, y_pred1 = K.minimum(y_true / 255, 1), K.minimum(y_pred / 255, 1)
    y_true1, y_pred1 = process_diag(y_true1), process_diag(y_pred1)
    tw = K.sum(K.cast(y_true1 * y_pred1, 'float32'), axis=[1, 2, 3])
    fw = K.sum(K.cast((1 - y_true1) * y_pred1, 'float32'), axis=[1, 2, 3])
    fb = K.sum(K.cast(y_true1 * (1 - y_pred1), 'float32'), axis=[1, 2, 3])
    prec = tw / (tw + fw + K.epsilon())
    rec =  tw / (tw + fb + K.epsilon())
    k1 = 1 -  K.abs(prec - rec) #penalty for huge difference in each picture
    k2 = 1 -  K.abs(K.mean(prec) - K.mean(rec)) #same, but generally for all dataset
    f1 = k1 * k2 * 2 * prec * rec / (prec + rec + K.epsilon())
    row_has_mps = K.cast(K.greater(K.sum(K.cast(y_true1 + tf.transpose(y_true1, [0, 2, 1, 3]), 'int64'), axis=[2]), 1), 'int64')
    col_has_mps = K.cast(K.greater(K.sum(K.cast(y_true1 + tf.transpose(y_true1, [0, 2, 1, 3]), 'int64'), axis=[1]), 1), 'int64')
    mask_col = tf.keras.layers.Multiply()([col_has_mps, K.cast(y_true1, 'int64')])
    mask_row = tf.transpose(tf.keras.layers.Multiply()([tf.transpose(K.cast(y_true1, 'int64'), [0, 2, 1, 3]), row_has_mps]), [0, 2, 1, 3])
    mask = K.cast(K.greater(mask_row +  mask_col, 0), 'float32')
    mps_total =  K.sum(mask)
    mps_detected = K.sum(mask * y_pred1)
    mps = (mps_detected + K.epsilon()) / (mps_total + K.epsilon())
    f1 *= mps
    return 1 - K.mean(f1)


def mps_metrics(y_true, y_pred): #metrics that returns the ammount of predicted multiplets
    y_true1, y_pred1 = K.minimum(y_true / 255, 1), K.minimum(y_pred / 255, 1)
    y_true1, y_pred1 = process_diag(y_true1), process_diag(y_pred1)
    row_has_mps = K.cast(K.greater(K.sum(K.cast(y_true1 + tf.transpose(y_true1, [0, 2, 1, 3]), 'int64'), axis=[2]), 1), 'int64')
    col_has_mps = K.cast(K.greater(K.sum(K.cast(y_true1 + tf.transpose(y_true1, [0, 2, 1, 3]), 'int64'), axis=[1]), 1), 'int64')
    mask_col = tf.keras.layers.Multiply()([col_has_mps, K.cast(y_true1, 'int64')])
    mask_row = tf.transpose(tf.keras.layers.Multiply()([tf.transpose(K.cast(y_true1, 'int64'), [0, 2, 1, 3]), row_has_mps]), [0, 2, 1, 3])
    mask = K.cast(K.greater(mask_row +  mask_col, 0), 'float32')
    mps_total =  K.sum(mask)
    mps_detected = K.sum(mask * y_pred1)
    return mps_detected

def f1_mps_metrics(y_true, y_pred): 
    return f1_metrics(y_true, y_pred) * mps_metrics(y_true, y_pred)

#Build model, generate data, launch training and testing
files = glob.glob(input_dir + '*.png') #remove old mirrored samples
for f in files:
    if '_mirror' in f:
        os.remove(f)
        os.remove(f.replace(input_dir, output_dir))

folds_num = 10
folds = split_files_kfold(folds_num)

for step in range(folds_num):
#define and compile model, all parametres here can be changed
    model = parallel_res_network(blocks_num=4, units_num=5, filters=[12, 10, 8, 6, 1], kernels=[13, 11, 9, 7, 5], activ='relu', bn=False, dr=0.1)
    model.compile(optimizer=Adagrad(lr=0.005), loss=f1_mps_loss, metrics=[precision, recall, mps_metrics, f1_mps_metrics])

    cl = CSVLogger(model_dir + 'training_' + str(step) + '.log')
    mc = ModelCheckpoint(model_dir + 'weights_' + str(step) + '.h5', save_best_only=True, monitor='val_f1_mps_metrics', mode='max')

    test_files = folds[step]
    train_files = []
    for i in range(folds_num):
        if i != step:
            train_files += folds[i]
    open(model_dir + 'datasets.txt', 'a').write(str(step) + ' step\ntrain:\n' + '\n'.join(train_files) + '\ntest:\n' + '\n'.join(test_files) + '\n')

#generate data and feed it into a compiled model

    train_size = calc_data_size(input_dir, train_files, train_bs, mirror=True, repeat=2) 
    test_size = calc_data_size(input_dir, test_files, 1)
    print('train files:', train_size, 'test files:', test_size)
    train_gen = generate_batches(input_dir, train_files, train_bs, mirror=True, repeat=2)
    test_gen = generate_batches(input_dir, test_files, 1)

    model.fit_generator(train_gen,
                        validation_data=test_gen,
                        steps_per_epoch=int(train_size/train_bs),
                        validation_steps=test_size,
                        epochs=epochs,
                        verbose=2,
                        shuffle=True,
                        callbacks=[cl, mc])

    print('Training done for ', step, ' of kfold')

#test model -- predict image for each sample and calculate precicion, recall and f1
    model = parallel_res_network(blocks_num=4, units_num=5, filters=[12, 10, 8, 6, 1], kernels=[13, 11, 9, 7, 5], activ='relu', bn=False, dr=0.1)
    model.load_weights(model_dir + 'weights_'  + str(step) + '.h5')
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
    print('Prediction done for ', step, ' of kfold')
    precs, recs, f1s = [], [], []
    prec, rec, f1 = 0, 0, 0 
    files_true = sorted(glob.glob(output_dir + '*.png'))
    files_pred = sorted(glob.glob(pred_dir + '*.png'))
    for f in files_pred:
        estimate(f.replace(pred_dir, output_dir), f)
    prec = round(sum(precs) / len(precs), 2)
    rec = round(sum(recs) / len(recs), 2)
    f1 = round(sum(f1s) / len(f1s), 2)
    print('Precision =', prec, '\nRecall =', rec, '\nF1 =', f1)
    open(model_dir + 'results.txt', 'a').write(str(step) + ' step\nprecicsion = ' + str(prec) + '\nrecall = ' + str(rec) + '\nf1 = ' + str(f1) + '\n')
    os.system('rm -r ' + pred_dir)
    files = glob.glob(input_dir + '*.png') #remove old mirrored samples
    for f in files:
        if '_mirror' in f:
            os.remove(f)
            os.remove(f.replace(input_dir, output_dir))
    K.clear_session()