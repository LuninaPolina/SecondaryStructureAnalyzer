'''Neural network output estimation by accuracy, precision, recall, f1 and levenstern distance'''

from PIL import Image
import numpy as np
from Bio import SeqIO
import glob
import csv
import matplotlib.pyplot as plt


in_dir_true = ''
in_dir_pred = ''
log_file = ''


dists = dict()
precs, recs, f1s = [], [], []


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


def get_dot(img):
    size = len(img)
    coords = []
    for i in range(size):
        for j in range(size):
            if img[i][j] == 255 and i != j:
                coords.append((min(i, j), max(i, j)))
    coords = set(coords)
    dot = ['.' for i in range(size)]
    for coord in coords:
        if dot[coord[0]] == '.':
            dot[coord[0]] = '('
        else:
            return 'err'
        if dot[coord[1]] == '.':
            dot[coord[1]] = ')'
        else:
            return 'err'
    return ''.join(dot)


def compare_dots(dot_true, dot_pred):
    if dot_pred == 'err':
        return 'err'
    n, m = len(dot_true), len(dot_pred)
    if n > m:
        dot_true, dot_pred = dot_pred, dot_true
        n, m = m, n
    current_row = range(n + 1)
    for i in range(1, m + 1):
        previous_row, current_row = current_row, [i] + [0] * n
        for j in range(1, n + 1):
            add, delete, change = previous_row[j] + 1, current_row[j - 1] + 1, previous_row[j - 1]
            if dot_true[j - 1] != dot_pred[i - 1]:
                change += 1
            current_row[j] = min(add, delete, change)
    return current_row[n]


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
        img_pred = binarize_output(np.array(img_pred)
        compare_images(img_true, img_pred)


cnt = 0
files_true = sorted(glob.glob(in_dir_true + '*.png'))
files_pred = sorted(glob.glob(in_dir_pred + '*.png'))
for f in files_pred:
    name = f.split('/')[-1].split('.')[0]
    estimate(f.replace(in_dir_pred, in_dir_true), f)
    cnt += 1
    if cnt % 100 == 0:
        print(cnt, 'done')
   

print('acc = ' + str(sum(accs) / len(accs)) + '\n')
print('prec = ' + str(sum(precs) / len(precs)) + '\n')
print('rec = ' + str(sum(recs) / len(recs)) + '\n')
print('fm = ' + str(sum(fmeras) / len(fmeras)) + '\n')
