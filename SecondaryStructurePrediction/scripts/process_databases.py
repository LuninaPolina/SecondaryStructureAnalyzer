import os
import glob
import random as rn
from PIL import Image
import numpy as np
import pandas as pd


tools_dir = ''
data_dir = ''
MIN_LEN, MAX_LEN = 0, 100
id_cnt = 0


def mkdir(dir):
    if not os.path.exists(dir):
        os.mkdir(dir)
 

def dot2img(seq, dot):
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
        print("Error")
        exit(0)
    img = Image.new('L', (size, size), (0)) 
    pixels = np.array(img)
    for el in mtrx:
        x, y = el
        pixels[x][y] = 255
    for i in range(len(seq)):
        pixels[i][i] = codes[seq[i]] 
    return Image.fromarray(pixels)


def cts2dbs(cts_dir, dbs_dir): 
    files = sorted(glob.glob(cts_dir + '*.ct'))
    rnastr_dir = tools_dir + 'RNAstructure/'
    for f in files:
        f2 = dbs_dir + f.split('/')[-1].split('.')[0] + '.db'
        command = 'export DATAPATH=' + rnastr_dir + 'data_tables/\n' + rnastr_dir + '/exe/ct2dot ' + f + ' 0 ' + f2 
        os.system(command)       
        

def from_rnastrand(src_dir, out_fasta, out_refdb):
    global id_cnt
    cts_dir = data_dir + 'cts/'
    mkdir(cts_dir)
    files = sorted(glob.glob(src_dir + '*.ct'))
    for f in files:
        f2 = cts_dir + f.split('/')[-1]
        with open(f, 'r') as inp, open(f2, 'w') as out:
            ct = inp.read().split('\n')
            seq_info = ''
            i = 0
            while ct[i][0] == '#':
                seq_info += ct[i]
                i += 1
            seq_info += ct[i] 
            seq_len = ct[-2].replace('\t', ' ').strip().split(' ')[0]
            new_ct = seq_len + ' ' + seq_info + '\n' + '\n'.join(ct[i + 1:])
            out.write(new_ct)
    dbs_dir = data_dir + 'dbs/'
    mkdir(dbs_dir)
    cts2dbs(cts_dir, dbs_dir)
    files = sorted(glob.glob(dbs_dir + '*.db'))
    with open(out_fasta, 'w') as out, open(out_refdb, 'a') as db:
        for f in files:
            try:
                data = open(f).read().split('\n')
            except:
                ()
            meta, seq, dot = data[0][1:], data[1].upper(), data[2]
            if all(s in 'ACGU' for s in seq) and not seq in seqs and len(seq) >= MIN_LEN and len(seq) <= MAX_LEN: 
                out.write('>' + str(id_cnt) + '\n' + seq + '\n' + dot + '\n')
                db.write(str(id_cnt) + '\t' + meta + '\t' + 'RNAstrand' + '\n')
                seqs.append(seq)
                id_cnt += 1
    os.system('rm -r ' + cts_dir)
    os.system('rm -r ' + dbs_dir)


def from_pseudobase(in_fasta, out_fasta, out_refdb):
    global id_cnt
    data = open(in_fasta).read().split('\n')
    with open(out_fasta, 'w') as out, open(out_refdb, 'a') as db:
        for i in range(0, len(data) - 2, 3):
            meta, seq, dot = data[i][2:], data[i + 1], data[i + 2]
            if all(s in 'ACGU' for s in seq) and not seq in seqs and len(seq) >= MIN_LEN and len(seq) <= MAX_LEN and not '~' in dot:
                out.write('>' + str(id_cnt) + '\n' + seq + '\n' + dot + '\n')
                db.write(str(id_cnt) + '\t' + meta + '\t' + 'Pseudobase' + '\n') 
                seqs.append(seq)
                id_cnt += 1


def from_rnacentral(in_fasta, out_fasta, ref_db):
    id_cnt = 0
    data = open(in_fasta).read().split('\n')
    with open(out_fasta, 'w') as out, open(out_refdb, 'a') as db:
        for i in range(0, len(data) - 2, 2):
            meta, seq = data[i][1:], data[i + 1]
            out.write('>' + str(id_cnt) + '\n' + seq.replace('T', 'U') + '\n')
            db.write(str(id_cnt) + '\t' + meta + '\t' + 'RNAcentral' + '\n') 
            id_cnt += 1
            

def from_gutell(src_dir, out_fasta, ref_db):
    global id_cnt
    dirs = glob.glob(src_dir + '*')
    seqs = []
    for d in dirs:
        cts_dir = data_dir + 'cts/'
        mkdir(cts_dir)
        files = sorted(glob.glob(d + '/*.ct'))
        for f in files:
            f2 = cts_dir + f.split('/')[-1]
            with open(f, 'r') as inp, open(f2, 'w') as out:
                ct = inp.read().split('\n')
                seq_info = ' '.join(ct[0:5])
                seq_len = ct[4].split(' ')[0]
                new_ct = seq_len + ' ' + seq_info + '\n' + '\n'.join(ct[5:])
                out.write(new_ct)
        dbs_dir = data_dir + 'dbs/'
        mkdir(dbs_dir)
        cts2dbs(cts_dir, dbs_dir)
        files = sorted(glob.glob(dbs_dir + '*.db'))
        with open(out_fasta, 'a') as out, open(out_refdb, 'a') as db:
            for f in files:
                try:
                    data = open(f).read().split('\n')
                except:
                    ()
                meta, seq, dot = data[0][1:], data[1].upper(), data[2]
                if all(s in 'ACGU' for s in seq) and not seq in seqs and len(seq) >= MIN_LEN and len(seq) <= MAX_LEN: 
                    out.write('>' + str(id_cnt) + '\n' + seq + '\n' + dot + '\n')
                    db.write(str(id_cnt) + '\t' + meta + '\t' + 'Gutell Lab CRW' + '\n')
                    seqs.append(seq)
                    id_cnt += 1
        os.system('rm -r ' + cts_dir)
        os.system('rm -r ' + dbs_dir)



def get_fasta_and_imgs(in_fasta, out_fasta, out_dir): 
    mkdir(out_dir)
    data = open(in_fasta).read().split('\n')
    with open(out_fasta, 'w') as out:
        for i in range(0, len(data) - 2, 3):
            meta, seq, dot = data[i][1:], data[i + 1], data[i + 2]  
            out.write('>' + meta + '\n' + seq + '\n')
            img = dot2img(seq, dot)
            img.save(out_dir + meta + '.png')
            

def get_imgs(in_fasta, out_dir): 
    mkdir(out_dir)
    data = open(in_fasta).read().split('\n')
    for i in range(0, len(data) - 2, 3):
        meta, seq, dot = data[i][1:], data[i + 1], data[i + 2]  
        if len(seq) <= 200:
            img = dot2img(seq, dot)
            img.save(out_dir + meta + '.png')
            

def process_all(inputs, out_fasta, out_refdb):
    seqs = []
    for (fasta, refdb) in inputs:
        with open(out_fasta, 'a') as out, open(out_refdb, 'a') as out_db:
            data = open(fasta).read().split('\n')
            db = pd.read_csv(refdb, sep='\t') 
            for i in range(0, len(data) - 2, 3):
                meta, seq, dot = data[i][1:], data[i + 1], data[i + 2]
                if not all(s in '.' for s in dot) and len(seq) <= 100:
                    if not seq in seqs:
                        seqs.append(seq)
                        out.write('>' + meta + '\n' + seq + '\n' + dot + '\n')
                        out_db.write(meta + '\t' + db.loc[db['id'] == int(meta)].values[0][1] + '\t' + db.loc[db['id'] == int(meta)].values[0][2] + '\t' + '\n')
                    