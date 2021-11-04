'''Functions for preparing input and reference data for neural network'''

from PIL import Image
import numpy as np
from Bio import SeqIO
import glob
import math
import os
from shutil import copyfile
from matplotlib import pyplot as plt


src_dir = ''
parsed_dir = src_dir + 'parsed/'
in_dir = src_dir + 'in/'
out_dir = src_dir + 'out/'
dot_file = src_dir + 'seq_dot.fasta'
seq_file = src_dir + 'seq.fasta'
bp_dir = src_dir + 'bp_lists/'
codes = {'A': 32, 'C': 64, 'G': 96, 'U': 128, 'T': 128}


def mkdir(dir):
    if not os.path.exists(dir):
        os.mkdir(dir)
 

def dot2img(seq, dot, diag=False):
    pairs = {")" : "(", "}" : "{", "]" : "[", ">" : "<", "a" : "A", "b" : "B"}
    codes = {'A': 32, 'C': 64, 'G': 96, 'U': 128, 'T': 128}
    stack = []
    mtrx = []
    size = len(seq)
    for i in range(len(dot)):
        if dot[i] == ".":
            continue
        else:
            paired = pairs.get(dot[i])
            if paired:
                j = len(stack) - 1
                while True:
                    if j < 0:
                        print("Pairing error!")
                        print(seq, dot)
                        exit(0)
                    if stack[j][1] == paired: 
                        mtrx.append((stack[j][0], i))
                        stack.pop(j)
                        break
                    else:
                        j-=1 
            else:
                stack.append((i, dot[i])) 
    if len(stack) != 0:
        print(seq, "Error")
        exit(0)
    img = Image.new('L', (size, size), (0)) 
    pixels = np.array(img)
    for el in mtrx:
        x, y = el
        pixels[x][y] = 255
    if diag:
       for i in range(len(seq)):
           pixels[i][i] = codes[seq[i]] 
    return Image.fromarray(pixels)


def seq_from_dot():
    data = open(dot_file).read().split('\n')
    with open(seq_file, 'w') as out:
        for i in range(0, len(data) - 2, 3):
            meta, seq, dot = data[i], data[i + 1], data[i + 2]
            out.write(meta + '\n' + seq + '\n')
            

def get_in_data(): #prepare input images from parsing output
    
    def add_stems_and_diag(img, seq):
        for i in range(len(img)):
            for j in range(i):
                if img[i][j] != 0:
                    img[i - 1][j + 1] = img[i][j]
                    img[i - 2][j + 2] = img[i][j]
        for i in range(len(seq)):
            img[i][i] = codes[seq[i]]
        return img

    def mirror(img, flag):
        for i in range(len(img)):
            for j in range(i):
                if flag == 'left':
                    img[j][i] = img[i][j]
                    img[i][j] = 0
                if flag == 'right':
                    img[i][j] = img[j][i]
                    img[j][i] = 0
        return img
    
    mkdir(in_dir)
    seqs = list(SeqIO.parse(open(seq_file), 'fasta'))
    files = glob.glob(parsed_dir + '*.bmp')
    for file in files:
        name = file.split('/')[-1].split('.')[0]
        seq = ''
        for sample in seqs:
            meta, s = str(sample.description), str(sample.seq)
            if name == meta:
                seq = s
                break
        if len(seq) == 0:
            print('err')
        img = np.array(Image.open(file).convert('L'))
        img = add_stems_and_diag(img, seq)
        img = mirror(img, 'left') 
        Image.fromarray(img).save(in_dir  + name + '.png')


def get_out_data(): #prepare reference images from dots
    mkdir(out_dir)
    data = open(dot_file).read().split('\n')
    for i in range(0, len(data) - 2, 3):
        meta, seq, dot = data[i][1:], data[i + 1], data[i + 2]  
        img = dot2img(seq, dot)
        img.save(out_dir + meta + '.png')
        

def get_out_data_pdb(diag=False): #prepare reference images from pdb database files
    mkdir(out_dir)
    files = glob.glob(bp_dir + '*.txt')
    seqs = list(SeqIO.parse(open(seq_file), 'fasta'))
    for file in files:
        name = file.split('/')[-1].split('.')[0]
        seq = ''
        for sample in seqs:
            meta, s = str(sample.description), str(sample.seq)
            if name == meta:
                seq = s
                break
        if len(seq) == 0:
            print('err')
        size = len(seq)
        img = np.array(Image.new('L', (size, size), (0)))
        data = open(file).read().split('\n')
        for line in data:
            if len(line) > 0:
                i, j = list(map(lambda x: int(x) - 1, line.split('_')))
                img[i][j] = 255
        if diag:
            for i in range(size):
                img[i][i] = codes[seq[i]] 
        Image.fromarray(img).save(out_dir + name + '.png')
        

def check_black(img):
    for i in range(len(img)):
        for j in range(len(img)):
            if i != j and img[i][j] == 255:
                return False
    return True


def delete_black():
    cnt = 0
    files = glob.glob(in_dir + '*.png')
    for f in files:
        f2 = f.replace('/in/', '/out/')
        img = np.array(Image.open(f))
        img2 = np.array(Image.open(f2))
        if check_black(img) or check_black(img2):
            cnt += 1
            os.system('rm ' + f)
            os.system('rm ' + f2)
    print(cnt)


def collect_set(in_dir, out_dir, max_len=200):
    files = glob.glob(in_dir  + '*.png')
    codes = {32: 'A', 64: 'C', 96: 'G', 128: 'U'}
    seqs = []
    for f_in in files:
        f_out = f_in.replace('/in/', '/out/')
        img_in = np.array(Image.open(f_in))
        img_out = np.array(Image.open(f_out))
        seq = ''
        for i in range(len(img_in)):
            seq += codes[img_in[i][i]]
        if len(img_in) <= max_len and not seq in seqs and not check_black(img_in) and not check_black(img_out):
            seqs.append(seq)
            name = f_in.split('/')[-1]
            for i in range(len(img_out)):
                img_out[i][i] = 0
            copyfile(f_in, out_dir + 'in/' + name)
            Image.fromarray(img_out).save(out_dir + 'out/' + name)