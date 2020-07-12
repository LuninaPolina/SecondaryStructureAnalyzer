from PIL import Image
import numpy as np
from Bio import SeqIO
import glob
import csv
import matplotlib.pyplot as plt

in_dir_true = '...'
in_dir_pred = '...'
log_file = '...'

tw, tb, fw, fb = 0, 0, 0, 0
dists = dict()
accs, precs, recs, fmeras = [], [], [], []

def binarize_output(img, coeff=0.75):
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
    global tw, tb, fw, fb
    local = [0, 0, 0, 0]
    size = len(img_true)
    for i in range(size):
        for j in range(size):
            if img_true[i][j] != img_pred[i][j] and i != j:
                if int(img_pred[i][j]) == 0:
                    fb += 1
                    local[3] += 1
                if int(img_pred[i][j]) == 255:
                    fw += 1
                    local[2] += 1
            elif img_true[i][j] == img_pred[i][j] and i != j:
		if int(img_true[i][j]) == 255:
                    tw += 1
                    local[0] += 1
		if int(img_true[i][j]) == 0:
                    tb += 1
                    local[1] += 1
    acc = (local[0] + local[1]) / (local[1] + local[0] + local[3] + local[2])
    prec = local[0] / (local[0] + local[2] + 0.00001)
    rec = local[0] / (local[0] + local[3] + 0.00001)
    fm = 2 * (prec * rec) / (prec + rec + 0.00001)
    accs.append(acc)
    precs.append(prec)
    recs.append(rec)
    fmeras.append(fm)
    
def estimate(f_true, f_pred):
    with Image.open(f_true) as img_true, Image.open(f_pred) as img_pred:
        img_true = np.array(img_true)
        img_pred = binarize_output(np.array(img_pred))
        name = f_true.split('/')[-1].split('.')[0]
        compare_images(img_true, img_pred)

        dot_pred = get_dot(img_pred)
        dot_true = get_dot(img_true)
        dots_dist = compare_dots(dot_true, dot_pred)
        if dots_dist in dists.keys():
            dists[dots_dist] += 1
        else:
            dists[dots_dist] = 1
    

cnt = 0
files_true = sorted(glob.glob(in_dir_true + '*.png'))
files_pred = sorted(glob.glob(in_dir_pred + '*.png'))
for i in range(len(files_pred)):
    estimate(files_true[i], files_pred[i])
    cnt += 1
    if cnt % 500 == 0:
        print(cnt, 'done')
    

with open(log_file, 'w') as log:
    acc = (tw + tb) / (tb + tw + fb + fw)
    prec = tw / (tw + fw)
    rec = tw / (tw + fb)
    fm = 2 * (prec * rec) / (prec + rec)
    log.write('For all contacts: \n')
    log.write('acc = ' + str(acc) + '\n')
    log.write('prec = ' + str(prec) + '\n')
    log.write('rec = ' + str(rec) + '\n')
    log.write('fm = ' + str(fm) + '\n')
    log.write('---------------------------------\n')
    log.write('For all images: \n')
    log.write('acc = ' + str(sum(accs) / len(accs)) + '\n')
    log.write('prec = ' + str(sum(precs) / len(precs)) + '\n')
    log.write('rec = ' + str(sum(recs) / len(recs)) + '\n')
    log.write('fm = ' + str(sum(fmeras) / len(fmeras)) + '\n')
    log.write('---------------------------------\n')
    log.write('dots dists percents:\n')
    if 'err' in dists.keys():
        err_persent = round(100 * dists['err'] / len(files_pred), 2)
        log.write('err: ' + str(err_persent) + '%\n')
        del dists['err']
    else:
        log.write('err: 0%\n')
    for k in sorted(dists.keys()):
        dist_persent = round(100 * dists[k] / len(files_pred), 2)
        log.write(str(k) + ': ' + str(dist_persent) + '%\n')





    






