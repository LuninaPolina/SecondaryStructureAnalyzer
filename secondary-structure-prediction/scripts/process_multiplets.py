'''Several algorithms for removing multiplets from neural network output'''

import numpy as np
import glob
from PIL import Image
from Bio import SeqIO
import os
import math
import time


def binarize_output(img, coeff=0.45):
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


def get_stem_len(i0, j0, img, only_bp=False):
    size = len(img)
    cnt = 1
    i, j = i0 + 1, j0 - 1
    while i < len(img) and j >= 0 and img[i][j] == 255 and (not only_bp or len(get_multiplets(i, j, img)) == 0):
        cnt += 1
        i += 1
        j -= 1
    i, j = i0 - 1, j0 + 1
    while i >= 0 and j < len(img) and img[i][j] == 255 and (not only_bp or len(get_multiplets(i, j, img)) == 0):
        cnt += 1
        i -= 1
        j += 1
    return(cnt)


def by_max_stem_len(img_pred): #for each multiplet remove all contacts except the one belonging to the longest stem
    size = len(img_pred)
    for i in range(size):
        for j in range(i + 1, size):
            if img_pred[i][j] == 255:
                mps = get_multiplets(i, j, img_pred)
                if len(mps) > 0:
                    max_stem_len = get_stem_len(i, j, img_pred, only_bp=True)
                    i_pick, j_pick = i, j
                    for (i0, j0) in mps:
                        stem_len = get_stem_len(i0, j0, img_pred, only_bp=True)
                        if stem_len > max_stem_len:
                            max_stem_len = stem_len
                            i_pick, j_pick = i0, j0
                    for (i0, j0) in mps:
                        if (i0, j0) != (i_pick, j_pick):
                            img_pred[i0][j0] = 0
    return img_pred


def by_min_stem_len(img_pred): #for each multiplet remove one contact belonging to the shortest stem, repeat until done
    size = len(img_pred)
    flag = True
    while flag:
        flag = False
        stems = dict()
        for i in range(size):
            for j in range(i + 1, size):
                if img_pred[i][j] == 255:
                    mps = get_multiplets(i, j, img_pred)
                    if len(mps) > 0:
                        flag = True
                        min_stem_len = get_stem_len(i, j, img_pred, only_bp=True)
                        i_pick, j_pick = i, j
                        for (i0, j0) in mps:
                            stem_len = get_stem_len(i0, j0, img_pred, only_bp=True)
                            if stem_len < min_stem_len:
                                min_stem_len = stem_len
                                i_pick, j_pick = i0, j0
                        img_pred[i_pick][j_pick] = 0
    return img_pred


def by_mps_size(img_pred, img_gray): #sort multiplets by size descending and then remove
    mps = dict()
    for i in range(size):
        for j in range(i + 1, size):
            if img_pred[i][j] == 255:
                mps0 = get_multiplets(i, j, img_pred)
                if len(mps0) > 0:
                    mps[(i, j)] = len(mps0)
    mps = {k: v for k, v in sorted(mps.items(), key=lambda item: -item[1])}
    for k in mps.keys():
        (i, j) = k[0], k[1]
        mps0 = get_multiplets(i, j, img_pred)
        if len(mps0) > 0:
            img_pred[i][j] = 0
    return img_pred


def by_stability(img_pred, seq): #sort multiplets by stability ascending, for each multiplet remove one contact with the lowest stability or belonging to the shortest stem, repeat until done
    size = len(img_pred)
    flag = True
    stability_range = ['CC', 'AC', 'UC', 'AA', 'UU', 'GA', 'GU', 'GG', 'AU', 'GC']
    bps = dict()
    for i in range(size):
        for j in range(i + 1, size):
            if img_pred[i][j] == 255 and len(get_multiplets(i, j, img_pred)) > 0:
                try:
                    stability = stability_range.index(seq[i] + seq[j])
                except:
                    stability = stability_range.index(seq[j] + seq[i])
                bps[(i, j)] = stability
    bps = {k: v for k, v in sorted(bps.items(), key=lambda item: item[1])}
    to_del = []
    while flag:
        flag = False
        for el in to_del:
            del bps[el]
        to_del = []
        for el in bps.keys():
            if not el in to_del:
                i, j = el[0], el[1]
                mps = get_multiplets(i, j, img_pred)
                if len(mps) > 0:
                    flag = True
                    i_pick, j_pick = i, j
                    try:
                        min_stability = stability_range.index(seq[i] + seq[j])
                    except:
                        min_stability = stability_range.index(seq[j] + seq[i])
                    for (i0, j0) in mps:
                        try:
                            stability = stability_range.index(seq[i0] + seq[j0])
                        except:
                            stability = stability_range.index(seq[j0] + seq[i0])
                        if stability < min_stability:
                            min_stability = stability
                            i_pick, j_pick = i0, j0
                        elif stability == min_stability:
                            stem_len_pick = get_stem_len(i_pick, j_pick, img_pred, only_bp=True)
                            stem_len0 = get_stem_len(i0, j0, img_pred, only_bp=True)
                            if stem_len0 < stem_len_pick:
                                i_pick, j_pick = i0, j0
                    img_pred[i_pick][j_pick] = 0
                    to_del.append((i_pick, j_pick))
    return img_pred


def by_min_stem_len_and_stability(img_pred, seq): #for each multiplet remove one contact with the lowest stability or belonging to the shortest stem, repeat until done
    size = len(img_pred)
    flag = True
    stability_range = ['CC', 'AC', 'UC', 'AA', 'UU', 'GA', 'GU', 'GG', 'AU', 'GC']
    while flag:
        flag = False
        min_stability = 11
        min_stem_len = math.inf
        for i in range(size):
            for j in range(i + 1, size):
                if img_pred[i][j] == 255:
                    mps = get_multiplets(i, j, img_pred)
                    if len(mps) > 0:
                        flag = True
                        for (i0, j0) in mps + [(i, j)]:
                            try:
                                stability = stability_range.index(seq[i0] + seq[j0])
                            except:
                                stability = stability_range.index(seq[j0] + seq[i0])
                            if stability < min_stability:
                                min_stability = stability
                                i_pick, j_pick = i0, j0
                            elif stability == min_stability:
                                stem_len_pick = get_stem_len(i_pick, j_pick, img_pred, only_bp=True)
                                stem_len0 = get_stem_len(i0, j0, img_pred, only_bp=True)
                                if stem_len0 < stem_len_pick:
                                    i_pick, j_pick = i0, j0
        if flag:
            img_pred[i_pick][j_pick] = 0
    return img_pred


def by_mps_size_and_stability(img_pred, seq): #sort multiplets by size descending, for each multiplet remove one contact with the lowest stability or belonging to the shortest stem, repeat until done
    size = len(img_pred)
    stability_range = ['CC', 'AC', 'UC', 'AA', 'UU', 'GA', 'GU', 'GG', 'AU', 'GC']
    mps_all = dict()
    for i in range(size):
        for j in range(i + 1, size):
            if img_pred[i][j] == 255:
                mps = get_multiplets(i, j, img_pred)
                if len(mps) > 0:
                    mps_all[(i, j)] = len(mps)
    mps_all = {k: v for k, v in sorted(mps_all.items(), key=lambda item: -item[1])}
    for k in mps_all.keys():
        (i, j) = k[0], k[1]
        if img_pred[i][j] == 255:
            mps = get_multiplets(i, j, img_pred)
            if len(mps) > 0:
                try:
                    min_stability = stability_range.index(seq[i] + seq[j])
                except:
                    min_stability = stability_range.index(seq[j] + seq[i])
                i_pick, j_pick = i, j
                for (i0, j0) in mps:
                    try:
                        stability = stability_range.index(seq[i0] + seq[j0])
                    except:
                        stability = stability_range.index(seq[j0] + seq[i0])
                    if stability < min_stability:
                        min_stability = stability
                        i_pick, j_pick = i0, j0
                    elif stability == min_stability:
                        min_stem_len = get_stem_len(i_pick, j_pick, img_pred, only_bp=True)
                        stem_len = get_stem_len(i0, j0, img_pred, only_bp=True)
                        if stem_len < min_stem_len:
                            i_pick, j_pick = i0, j0
                img_pred[i_pick][j_pick] = 0
    return img_pred


def by_probability(img_pred, img_gray): #for each multiplet remove one contact having the darkest pixel, repeat until done
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
                    min_prob = img_gray[i][j]
                    i_pick, j_pick = i, j
                    for (i0, j0) in mps:
                        if img_gray[i0][j0] < min_prob:
                            i_pick, j_pick = i0, j0
                            min_prob = img_gray[i0][j0]
                    img_pred[i_pick][j_pick] = 0
                    to_delete.append((i_pick, j_pick))
                else:
                    to_delete.append((i, j))
    return img_pred


#current best
def by_mps_size_and_min_stem_len(img_pred): #sort multiplets by size descending, for each multiplet remove one contact belonging to the shortest stem, repeat until done
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
                    min_stem_len = get_stem_len(i, j, img_pred, only_bp=True)
                    i_pick, j_pick = i, j
                    for (i0, j0) in mps:
                        stem_len = get_stem_len(i0, j0, img_pred, only_bp=True)
                        if stem_len < min_stem_len:
                            i_pick, j_pick = i0, j0
                            min_stem_len = stem_len
                    img_pred[i_pick][j_pick] = 0
                    to_delete.append((i_pick, j_pick))
                else:
                    to_delete.append((i, j))
    return img_pred


def check_multiplets(img_pred):
    size = len(img_pred)
    for i in range(size):
        for j in range(i + 1, size):
            if img_pred[i][j] == 255:
                mps = get_multiplets(i, j, img_pred)
                if len(mps) > 0:
                    return True
    return False


def compare_images(img_true, img_pred):
    size = len(img_true)
    tw, fw, fb = 0, 0, 0 
    img_pred = by_mps_size_and_min_stem_len(img_pred)
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


start = time.time()
fms, precs, recs = [] , [], []     
in_dir = ''
out_dir = ''
files = glob.glob(in_dir + '*.png')
for f_pred in files:
    img_gray = np.array(Image.open(f_pred))
    f_true = f_pred.replace(in_dir, out_dir)
    img_true = np.array(Image.open(f_true))
    img_pred = binarize_output(img_gray, 0.6)
    prec, rec, fm = compare_images(img_true, img_pred)
    fms.append(fm)
    precs.append(prec)
    recs.append(rec)
print(round(sum(precs) / len(precs), 2), round(sum(recs) / len(recs), 2), round(sum(fms) / len(fms), 2))  
print(time.time() - start)
    