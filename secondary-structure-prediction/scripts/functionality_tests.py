'''Many tests that estimate models functionality by different criteria'''

from PIL import Image
import numpy as np
import glob
import os
import math
import time 
from shutil import copyfile
import statistics


#Data processing functions


brs = [("(", ")"), ("[", "]"), ("{", "}"), ("<", ">"), ("a", "A"), ("b", "B"), ("c", "C"), ("d", "D"), ("e", "E"), ("f", "F")]
val_ids_rs = '' #should be a string of ids separated by spaces


def binarize_output(img, coeff=0.6):
    size = len(img)
    for i in range(size):
        for j in range(size):
            if i < j:
                if img[i][j] > 255 * coeff:
                    img[i][j] = 255
                else:
                    img[i][j] = 0
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


def get_stem_len(i0, j0, img, only_bp=True):
    size = len(img)
    stem_len = 1
    i, j = i0 + 1, j0 - 1
    while i < len(img) and j >= 0 and img[i][j] == 255 and (not only_bp or len(get_multiplets(i, j, img)) == 0):
        stem_len += 1
        i += 1
        j -= 1
    i, j = i0 - 1, j0 + 1
    while i >= 0 and j < len(img) and img[i][j] == 255 and (not only_bp or len(get_multiplets(i, j, img)) == 0):
        stem_len += 1
        i -= 1
        j += 1
    return(stem_len)


def get_mps(img):
    size = len(img)
    mps_all = []
    for i in range(size):
        for j in range(i + 1, size):
            if img[i][j] == 255:
                mps = get_multiplets(i, j, img) + [(i, j)]
                if len(mps) > 1:
                    exists = False
                    for m in mps_all:
                        if len(set(m) ^ set(mps)) == 0:
                            exists = True
                            break
                    if not exists:
                        mps_all.append(mps) 
    to_del, mps_all2 = [], []
    idx = -1
    for i in range(len(mps_all)):
        if not mps_all[i] in to_del:
            mps_all2.append(mps_all[i])
            idx += 1
            state_change = True                 
            while state_change: 
                state_change = False
                for j in range(i + 1, len(mps_all)):
                    if not mps_all[j] in to_del:
                        for m1 in mps_all[j]:
                            for m2 in mps_all2[idx]:
                                if not mps_all[j] in to_del:
                                    if m1 == m2:
                                        mps_all2[idx] += mps_all[j]
                                        state_change = True
                                        to_del.append(mps_all[j])   
            
    return list(map(lambda x: list(set(x)), mps_all2))


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


def remove_multiplets_min_stem(img_pred):
    size = len(img_pred)
    mps_nums = dict()
    for i in range(size):
        for j in range(i + 1, size):
            if img_pred[i][j] == 255:
                mps = get_multiplets(i, j, img_pred)
                if len(mps) > 0:
                    mps_nums[(i, j)] = len(mps)
    mps_nums = {k: v for k, v in sorted(mps_nums.items(), key=lambda item: -item[1])}
    to_delete = []
    while len(mps_nums) > 0:
        for el in to_delete:
            del mps_nums[el]
        to_delete = []
        for (i, j) in mps_nums.keys():
            if not (i, j) in to_delete:
                mps = get_multiplets(i, j, img_pred)
                if len(mps) > 0:
                    min_stem_len = get_stem_len(i, j, img_pred)
                    i_pick, j_pick = i, j
                    for (i0, j0) in mps:
                        stem_len = get_stem_len(i0, j0, img_pred)
                        if stem_len < min_stem_len:
                            i_pick, j_pick = i0, j0
                            min_stem_len = stem_len
                    img_pred[i_pick][j_pick] = 0
                    to_delete.append((i_pick, j_pick))
                else:
                    to_delete.append((i, j))
    return img_pred


def compare_images(img_true, img_pred):
    size = len(img_true)
    tw, fw, fb = 0, 0, 0 
    for i in range(size):
        for j in range(i + 1, size):  
            if img_true[i][j] != img_pred[i][j] and i != j:
                if int(img_pred[i][j]) == 0:
                    fb += 1
                else:
                    fw += 1
            elif img_true[i][j] == img_pred[i][j] and i != j:
                if int(img_true[i][j]) == 255:
                    tw += 1
    prec = tw / (tw + fw + 0.00001)
    rec = tw / (tw + fb + 0.00001)
    fm = 2 * (prec * rec) / (prec + rec + 0.00001)
    return prec, rec, fm


def get_pk_pairs(img):
    size = len(img)
    coords, pk_pairs = [], []
    for i in range(size):
        for j in range(i + 1, size):
            if img[i][j] == 255:
                coords.append((min(i, j), max(i, j)))
    for c1 in coords:
        for c2 in coords:
            if (c1[0] > c2[0] and c1[0] < c2[1]) and (c1[1] < c2[0] or c1[1] > c2[1]):
                pk_pairs.append(c1)
                pk_pairs.append(c2)
    return list(set(pk_pairs))  


def get_pknots(img):
    size = len(img)
    coords, pk_pairs, pknots = [], [], []
    for i in range(size):
        for j in range(i + 1, size):
            if img[i][j] == 255:
                coords.append((min(i, j), max(i, j)))
    for c1 in coords:
        for c2 in coords:
            if (c1[0] > c2[0] and c1[0] < c2[1]) and (c1[1] < c2[0] or c1[1] > c2[1]):
                if not (c1, c2) in pk_pairs and not (c2, c1) in pk_pairs:
                    pk_pairs.append((c1, c2))
    to_del = []
    idx = -1
    for i in range(len(pk_pairs)):
        if not pk_pairs[i] in to_del:
            pknots.append([pk_pairs[i]])
            idx += 1
            state_change = True                 
            while state_change: 
                state_change = False
                for j in range(i + 1, len(pk_pairs)):
                    if not pk_pairs[j] in to_del:
                        c1, c2 = pk_pairs[j][0], pk_pairs[j][1]
                        for (c3, c4) in pknots[idx]:
                            if not (c1, c2) in to_del:
                                if c1 == c3 or c1 == c4 or c2 == c3 or c2 == c4:
                                    pknots[idx].append(pk_pairs[j])
                                    state_change = True
                                    to_del.append(pk_pairs[j])   
    pknots2 = []
    for i in range(len(pknots)):
        pknots2.append([])
        for pair in pknots[i]:
            for c in pair:
                if not c in pknots2[i]:
                    pknots2[i].append(c)      
    return pknots2


#Test functions


def f1(in_dir, out_dir, val_ids): #classical precision, recall and f1                  
    files = glob.glob(in_dir + '*.png')
    precs, recs, fms = [], [], []
    cnt = 0
    for f_pred in files:
         if f_pred.split('/')[-1].split('.')[0] in val_ids.split(' '):
            f_true = f_pred.replace(in_dir, out_dir)
            img_true = np.array(Image.open(f_true))
            img_pred = np.array(Image.open(f_pred))
            img_pred = binarize_output(img_pred)
            img_pred = remove_multiplets_min_stem(img_pred)
            prec, rec, fm = compare_images(img_true, img_pred)
            fms.append(fm)
            precs.append(prec)
            recs.append(rec)
            cnt += 1
    print(cnt, 'prec =', round(100 * sum(precs) / len(precs)), 'rec =', round(100 * sum(recs) / len(recs)), 'fm =', round(100 * sum(fms) / len(fms)))  


def f1_pks_images(in_dir, out_dir, val_ids): #precision, recall and f1 only among the pseudoknotted structures                
    files = glob.glob(in_dir + '*.png')
    cnt_pk, cnt = 0, 0
    precs, recs, fms = [], [], []
    for f_pred in files:
        if f_pred.split('/')[-1].split('.')[0] in val_ids.split(' '):
            cnt += 1
            f_true = f_pred.replace(in_dir, out_dir)
            img_true = np.array(Image.open(f_true))
            img_pred = np.array(Image.open(f_pred))
            img_pred = binarize_output(img_pred) 
            img_pred = remove_multiplets_min_stem(img_pred)
            pks_true = get_pk_pairs(img_true)
            if len(pks_true) > 0:
                cnt_pk += 1
                prec, rec, fm = compare_images(img_true, img_pred)
                fms.append(fm)
                precs.append(prec)
                recs.append(rec)
    print(cnt_pk, 'samples with pseudoknots', cnt, 'total samples')
    print('prec =', round(100 * sum(precs) / len(precs)), 'rec =', round(100 * sum(recs) / len(recs)), 'fm =', round(100 * sum(fms) / len(fms)))  


def f1_pks_contacts(in_dir, out_dir, val_ids): #precision, recall and f1 only among pseudoknotted contacts                    
    files = glob.glob(in_dir + '*.png')
    tw, fw, fb = 0, 0, 0
    cnt_pk, cnt = 0, 0
    for f_pred in files:
        if f_pred.split('/')[-1].split('.')[0] in val_ids.split(' '):
            f_true = f_pred.replace(in_dir, out_dir)
            img_true = np.array(Image.open(f_true))
            img_pred = np.array(Image.open(f_pred))
            img_pred = binarize_output(img_pred) 
            img_pred = remove_multiplets_min_stem(img_pred)
            pks_true = get_pk_pairs(img_true)
            pks_pred = get_pk_pairs(img_pred)
            for i in range(len(img_true)):
                for j in range(i + 1, len(img_true)):
                    if img_true[i][j] == 255:
                        cnt += 1
                        if (i, j) in pks_true:
                            cnt_pk += 1
                            if img_pred[i][j] == 255:
                                tw += 1
                            if img_pred[i][j] == 0:
                                fb += 1
                    elif (i, j) in pks_pred:
                        fw += 1
    prec = round(100 * tw / (tw + fw))
    rec = round(100 * tw / (tw + fb))
    fm = round(2 * (prec * rec) / (prec + rec))
    print(cnt_pk, 'pseudoknots contacts', cnt, 'total contacts')
    print('prec =', prec, 'rec =', rec, 'fm =', fm)


def pks_detected(in_dir, out_dir, val_ids, error=0): #the amount of correctly guessed pseudoknots with maximum allowed error                
    files = glob.glob(in_dir + '*.png')
    true_pk, cnt_pk = 0, 0
    for f_pred in files:
        if f_pred.split('/')[-1].split('.')[0] in val_ids.split(' '):
            f_true = f_pred.replace(in_dir, out_dir)
            img_true = np.array(Image.open(f_true))
            img_pred = np.array(Image.open(f_pred))
            img_pred = binarize_output(img_pred)
            img_pred = remove_multiplets_min_stem(img_pred)
            pknots_true = get_pknots(img_true)
            pknots_pred = get_pknots(img_pred)
            cnt_pk += len(pknots_true)
            for pk1 in pknots_true:
                for pk2 in pknots_pred:
                    pk1, pk2 = list(set(pk1)), list(set(pk2))
                    if (len(list(filter(lambda x: not x in pk1, pk2))) + len(list(filter(lambda x: not x in pk2, pk1)))) / len(pk1) <= error:
                        true_pk += 1
                        break
    print(cnt_pk, 'pseudoknots')
    print('true_pk =', true_pk)


def bps_detected(in_dir, out_dir, val_ids): #the amounts of correctly guessed Watson-Crick, G-U and other wobble base pairs
    codes = {32: 'A', 64: 'C', 96: 'G', 128: 'U'}
    canon_bps = ['AU', 'UA', 'CG', 'GC']
    wobble_bps = ['GU', 'UG']
    files = glob.glob(in_dir + '*.png')
    cnt_nc, cnt_c, cnt_w = 0, 0, 0
    true_w, true_c, true_nc = 0, 0, 0
    for f_pred in files:
        if f_pred.split('/')[-1].split('.')[0] in val_ids.split(' '):
            f_true = f_pred.replace(in_dir, out_dir)
            f_in = f_true.replace('/out/', '/in/')
            img_true = np.array(Image.open(f_true))
            img_in = np.array(Image.open(f_in))
            img_pred = np.array(Image.open(f_pred))
            img_pred = binarize_output(img_pred)
            img_pred = remove_multiplets_min_stem(img_pred)
            seq = ''
            for i in range(len(img_in)):
                seq += codes[img_in[i][i]]
            for i in range(len(img_true)):
                for j in range(i + 1, len(img_true)):
                    if img_true[i][j] == 255:
                        if seq[i] + seq[j] in canon_bps:
                            cnt_c += 1
                            if img_pred[i][j] == 255:
                                true_c += 1
                        elif seq[i] + seq[j] in wobble_bps:
                            cnt_w += 1
                            if img_pred[i][j] == 255:
                                true_w += 1
                        else:
                            cnt_nc += 1
                            if img_pred[i][j] == 255:
                                true_nc += 1
    print(cnt_c, 'total canon contacts', true_c, 'predicted canon contacts')
    print(cnt_w, 'total wobble contacts', true_w, 'predicted wobble contacts')
    print(cnt_nc, 'total non-canon contacts', true_nc, 'predicted non-canon contacts')


def mps_pairs_detected(in_dir, out_dir, val_ids): #the amount of correctly guessed multiplets (as connected pairs of base-pairs)
    files = sorted(glob.glob(in_dir + '*.png'))
    cnt_mps, cnt_true_mps, cnt_cont, cnt_true_cont = 0, 0, 0, 0
    ids = ''
    for f_pred in files:
        if f_pred.split('/')[-1].split('.')[0] in val_ids.split(' '): 
            f_true = f_pred.replace(in_dir, out_dir)
            img_true = np.array(Image.open(f_true))
            img_pred = np.array(Image.open(f_pred))
            img_pred = binarize_output(img_pred, 1)
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


def mps_detected(in_dir, out_dir, val_ids, mps_size, error): #the amount of correctly guessed multiplets of fixed size with maximum allowed error                      
    files = glob.glob(in_dir + '*.png')
    true_mps, cnt_mps = 0, 0
    for f_pred in files:
        if f_pred.split('/')[-1].split('.')[0] in val_ids.split(' '):
            f_true = f_pred.replace(in_dir, out_dir)
            img_true = np.array(Image.open(f_true))
            img_pred = np.array(Image.open(f_pred))
            img_pred = binarize_output(img_pred)
            mps_true = get_mps(img_true)
            mps_pred = get_mps(img_pred)
            for m1 in mps_true:
                if len(m1) == mps_size:
                    cnt_mps += 1
                    for m2 in mps_pred:
                        if (len(list(filter(lambda x: not x in m1, m2))) + len(list(filter(lambda x: not x in m2, m1)))) / len(m1) <= error:
                            true_mps += 1
                            break
    print(cnt_mps, 'multiplets')
    print('true_mps =', true_mps)