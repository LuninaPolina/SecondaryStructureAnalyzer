import os
from Bio import SeqIO
from PIL import Image
import numpy as np
import glob
import time
import pandas as pd


tools_dir = ''
data_dir = ''
rs_dir = ''


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


def from_pknotsrg(fasta_file, out_dir):
    seqs = list(SeqIO.parse(open(fasta_file), 'fasta'))
    out_file = data_dir + 'pknotsRG/out.txt'
    for sample in seqs:
        meta, seq = str(sample.description), str(sample.seq)
        open(out_file, 'a').write('>' + meta + '\n')
        command = tools_dir + 'pknotsRG/pknotsRG-mfe ' + seq + ' >>' + out_file
        os.system(command) 
    output = open(out_file).read().split('\n')
    mkdir(out_dir)
    for i in range(0, len(output) - 2, 3):
        meta, seq, dot = output[i][1:], output[i + 1], output[i + 2].split(' ')[0]
        img = dot2img(seq, dot)
        img.save(out_dir + meta + '.png')
    os.remove(out_file)
    

def from_hotknots(fasta_file, out_dir):
    seqs = list(SeqIO.parse(open(fasta_file), 'fasta'))
    out_file = data_dir + 'hotknots/out.txt'
    for sample in seqs:
        meta, seq = str(sample.description), str(sample.seq)
        open(out_file, 'a').write('>' + meta + '\n')
        command = 'cd ' + tools_dir + 'hotknots/bin/\n' + './HotKnots -noPS -b -s ' + seq + ' >>' + out_file
        os.system(command)    
        open(out_file, 'a').write('end_sample\n')
    output = open(out_file).read().split('end_sample\n')[:-1]
    mkdir(out_dir)
    for line in output:
        data = line.split('\n')
        try:
            meta, seq, dot = data[0][1:], data[2][5:], data[3][5:].split('\t')[0]
        except:
            print(line)
        img = dot2img(seq, dot)
        img.save(out_dir + meta + '.png')
    os.remove(out_file)


def from_e2efold(step, fasta_file, out_dir):
    #change test folder and save folder in config.json
    if step == 1: #create .seq file for each seq
        seqs = list(SeqIO.parse(open(fasta_file), 'fasta'))
        seqs_dir = data_dir + 'e2efold/seqs/'
        os.mkdir(seqs_dir)
        for sample in seqs:
            meta, seq = str(sample.description), str(sample.seq) 
            open(seqs_dir + meta, 'w').write(seq)
    #run model from termunal
    if step == 2:
        cts_dir = data_dir + 'e2efold/cts/'
        dbs_dir = data_dir + 'e2efold/dbs/'
        os.mkdir(dbs_dir)
        cts2dbs(cts_dir, dbs_dir)
        files = glob.glob(dbs_dir + '*.db')
        for f in files:
            data = open(f).read().split('\n')
            meta, seq, dot = data[0][1:-3], data[1], data[2]
            img = dot2img(seq, dot)
            img.save(out_dir + meta + '.png')
    
def from_spotrna(fasta_file, out_dir):
    tmp_dir = data_dir + 'SPOT-RNA/out/'
    mkdir(tmp_dir)
    print('In terminal print: ')
    print('cd ' + tools_dir + 'SPOT-RNA/')
    print('conda activate venv')
    print('python3 SPOT-RNA.py  --inputs ' + fasta_file + ' --outputs \'' + tmp_dir + '\'' + ' --gpu 0')
    input("Press Enter when finished...")
    cts2dbs(tmp_dir, tmp_dir)
    files = glob.glob(tmp_dir + '*.db')
    mkdir(out_dir)
    for f in files:
        data = open(f).read().split('\n')
        meta, seq, dot = data[0].split('\t')[0][1:], data[1], data[2]
        img = dot2img(seq, dot)
        img.save(out_dir + meta + '.png')
    os.system('rm -r ' + tmp_dir)
        
        
def from_knotty(fasta_file, out_dir):
    seqs = list(SeqIO.parse(open(fasta_file), 'fasta'))
    out_file = data_dir + 'Knotty/out.txt'
    for sample in seqs:
        meta, seq = str(sample.description), str(sample.seq)
        if len(seq) <= 100:
            open(out_file, 'a').write('>' + meta + '\n')
            command = 'cd ' + tools_dir + 'Knotty\n' + './knotty ' + seq + ' >>' + out_file
            os.system(command) 
    output = open(out_file).read().split('\n')
    mkdir(out_dir)
    for i in range(0, len(output) - 3, 3):
        meta, seq, dot = output[i][1:], output[i + 1].split(' ')[1], output[i + 2].split(' ')[1]
        img = dot2img(seq, dot)
        img.save(out_dir + meta + '.png')
    os.remove(out_file)


def from_ipknot(fasta_file, out_dir):
    seqs = list(SeqIO.parse(open(fasta_file), 'fasta'))
    seqs_dir = data_dir + 'ipknot/seqs/'
    mkdir(seqs_dir)
    out_file = data_dir + 'ipknot/out.txt'
    for sample in seqs:
        meta, seq = str(sample.description), str(sample.seq) 
        open(seqs_dir + meta, 'w').write('>' + meta + '\n' + seq)
        command = 'cd ' + tools_dir + 'ipknot\n' + 'ipknot ' + seqs_dir + meta + ' >>' + out_file
        os.system(command)
    output = open(out_file).read().split('Long-step dual simplex will be used')[1:]
    mkdir(out_dir)
    for el in output:
        if not 'Error' in el:
            el = el.split('\n')
            meta, seq, dot = el[1][1:], el[2], el[3]
            img = dot2img(seq, dot)
            img.save(out_dir + meta + '.png')
    os.remove(out_file)
    os.system('rm -r ' + seqs_dir)

            
def from_rnastructure(fasta_file, out_dir):
    seqs = list(SeqIO.parse(open(fasta_file), 'fasta'))
    seqs_dir = data_dir + 'RNAstructure/seqs/'
    cts_dir = data_dir + 'RNAstructure/cts/'
    dbs_dir = data_dir + 'RNAstructure/dbs/'
    mkdir(seqs_dir)
    mkdir(cts_dir)
    mkdir(dbs_dir)
    for sample in seqs:
        meta, seq = str(sample.description), str(sample.seq) 
        open(seqs_dir + meta, 'w').write(seq)
        command = 'export DATAPATH=' + tools_dir  + 'RNAstructure/data_tables/\n' + tools_dir + 'RNAstructure/exe/ProbKnot ' + seqs_dir + meta + ' ' + cts_dir + meta + '.ct --sequence'
        os.system(command)
    cts2dbs(cts_dir, dbs_dir)
    files = glob.glob(dbs_dir + '*.db')
    mkdir(out_dir)
    for f in files:
        data = open(f).read().split('\n')
        meta, seq, dot = data[0][1:], data[1], data[2]
        img = dot2img(seq, dot)
        img.save(out_dir + meta + '.png')
    os.system('rm -r ' + seqs_dir)
    os.system('rm -r ' + cts_dir)
    os.system('rm -r ' + dbs_dir)
 

def from_centroid(fasta, out_dir): # --noncanonical 
    out_file = data_dir + 'centroid-rna-package/out.txt'
    command = 'cd ' + tools_dir + 'centroid-rna-package/build/src/\n' + './centroid_fold -o ' + out_file + ' ' + fasta
    os.system(command)  
    output = open(out_file).read().split('\n')
    mkdir(out_dir)
    for i in range(0, len(output) - 3, 3):
        meta, seq, dot = output[i][1:], output[i + 1], output[i + 2].split(' ')[0]
        img = dot2img(seq, dot)
        img.save(out_dir + meta + '.png')
    os.remove(out_file)

    
def calculate_fmera(true_dir, pred_dir, db_file):
    db = pd.read_csv(db_file, sep='\t')
    files_true = sorted(glob.glob(true_dir + '*.png'))
    files_pred = sorted(glob.glob(pred_dir + '*.png'))
    precs, recs, fmeras = [], [], []
    for f in files_pred:
        db_name = db.loc[db['id'] == int(f.split('/')[-1].split('.')[0])].values[0][2]
        img_true = np.array(Image.open(f.replace(pred_dir, true_dir)))
        img_pred = np.array(Image.open(f))
        if len(img_true) <= 100 and 'RNAstrand' in db_name:
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
            prec = tw/ (tw + fw + 0.00001)
            rec = tw / (tw + fb + 0.00001)
            fm = 2 * (prec * rec) / (prec + rec + 0.00001)
            precs.append(prec)
            recs.append(rec)
            fmeras.append(fm)
    print('prec =', str(sum(precs) / len(precs)))
    print('rec =', str(sum(recs) / len(recs)))
    print('fm =', str(sum(fmeras) / len(fmeras)))


def rm_black(in_dir, out_dir):
    def check_black(img):
        for i in range(len(img)):
            for j in range(len(img)):
                if img[i][j] != 0 and i != j:
                    return False
        return True
    files_in = glob.glob(in_dir + '*.png')
    files_out = glob.glob(out_dir + '*.png')
    for f_out in files_out:
        f_in = f_out.replace(out_dir, in_dir)
        if not f_in in files_in:
            os.remove(f_out)
        else:
            img = np.array(Image.open(f_out))
            if check_black(img):
                os.remove(f_out)