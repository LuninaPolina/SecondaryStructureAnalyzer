'''Parsing result to neural network input transormation functions'''

from PIL import Image
import numpy as np
from Bio import SeqIO
import glob


in_dir = ''
out_dir = ''
seq_file = ''
codes = {'A': 32, 'C': 64, 'G': 96, 'U': 128, 'T': 128}


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


def add_borders(img, size):
    img_new = Image.new('L', (size, size), (0))
    pixels = np.array(img)
    pixels_new = np.array(img_new)
    for i in range(len(pixels)):
        for j in range(len(pixels)):
            pixels_new[i][j] = pixels[i][j]
    return pixels_new

seqs = list(SeqIO.parse(open(seq_file), 'fasta'))
files = glob.glob(in_dir + '*.bmp')
for file in files:
    name = file.split('/')[-1].split('.')[0]
    seq = ''
    for sample in seqs:
        meta, seq = str(sample.description), str(sample.seq)
        if name == meta:
            break
    if len(seq) == 0:
        print('err')
    img = np.array(Image.open(file).convert('L'))
    img = add_stems_and_diag(img, seq)
    #img = add_borders(mirror(img, 'left'), size)
    img = mirror(img, 'left') #when different sizes
    Image.fromarray(img).save(out_dir  + name + '.png')
