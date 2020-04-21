from PIL import Image, ImageOps
import numpy as np
from bitstring import BitArray
import csv
from Bio import SeqIO
import glob

#change here
files = ['../train.csv',
         '../valid.csv',
         '../test.csv']
out_dir = '../src/'
out_dir2 = '../images80/'
db_file = '../ref_db.csv'
seq_file = '../all_seq.fasta'


def vec2img(vec, length, out_dir):
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
    img.save(out_dir + vec[0][1:] + '.bmp')

def process_file(inp_file, out_dir, length):
    with open(inp_file, 'r') as inp:
        data = csv.reader(inp)
        for line in data:
            vec2img(line, length, out_dir)

def crop(img_file, out_dir, size):
    img = Image.open(img_file)
    i_d = img_file.split('\\')[-1].split('.')[0]
    seqs = SeqIO.parse(open(seq_file), 'fasta')         
    for sample in seqs:
        meta, seq = sample.description, str(sample.seq)
        if meta == i_d:
            ln = len(seq.split('D')[0])
            break
    img_crop = Image.new('RGB', (ln, ln), (255,255,255))
    img_crop.paste(img)
    img_crop = img_crop.resize((size, size))
    img_crop.save(out_dir + i_d + '.bmp')
    

for file in files:
    process_file(file, out_dir, 220)
files2 = [f for f in glob.glob(out_dir + '*.bmp')]
for file in files2:
    crop(file, out_dir2, 80)
    
    




