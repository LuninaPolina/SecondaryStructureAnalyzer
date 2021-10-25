import os
from Bio import SeqIO
from PIL import Image
import numpy as np
import glob
import random as rn
import time


data_dir = ''
tools_dir = ''


def mkdir(dir):
    if not os.path.exists(dir):
        os.mkdir(dir)


def from_pknotsrg(fasta_file):
    seqs = list(SeqIO.parse(open(fasta_file), 'fasta'))
    out_file = data_dir + 'out_pknots.txt'
    for sample in seqs:
        meta, seq = str(sample.description), str(sample.seq)
        open(out_file, 'a').write('>' + meta + '\n')
        command = tools_dir + 'pknotsRG/pknotsRG-mfe ' + seq + ' >>' + out_file
        os.system(command) 
    

def from_hotknots(fasta_file):
    seqs = list(SeqIO.parse(open(fasta_file), 'fasta'))
    out_file = data_dir + 'out_hot.txt'
    for sample in seqs:
        meta, seq = str(sample.description), str(sample.seq)
        open(out_file, 'a').write('>' + meta + '\n')
        command = 'cd ' + tools_dir + 'hotknots/bin/\n' + './HotKnots -noPS -b -s ' + seq + ' >>' + out_file
        os.system(command)    
    

def from_spotrna(fasta_file):
    tmp_dir = data_dir + 'SPOT-RNA/'
    mkdir(tmp_dir)
    command = 'cd ' + tools_dir + 'SPOT-RNA/ \nconda run -n venv python3 SPOT-RNA.py  --inputs ' + fasta_file + ' --outputs \'' + tmp_dir + '\'' + ' --gpu 0' 
    os.system(command)
    

def from_ipknot(fasta_file):
    seqs = list(SeqIO.parse(open(fasta_file), 'fasta'))
    seqs_dir = data_dir + 'seqs_ip/'
    mkdir(seqs_dir)
    out_file = data_dir + 'out_ip.txt'
    for sample in seqs:
        meta, seq = str(sample.description), str(sample.seq) 
        open(seqs_dir + meta, 'w').write('>' + meta + '\n' + seq)
        command = 'cd ' + tools_dir + 'ipknot\n' + 'ipknot ' + seqs_dir + meta + ' >>' + out_file
        os.system(command)
    

def from_rnastructure(fasta_file):
    seqs = list(SeqIO.parse(open(fasta_file), 'fasta'))
    seqs_dir = data_dir + 'seqs_rs/'
    cts_dir = data_dir + 'cts_rs/'
    mkdir(seqs_dir)
    mkdir(cts_dir)
    for sample in seqs:
        meta, seq = str(sample.description), str(sample.seq) 
        open(seqs_dir + meta, 'w').write(seq)
        command = 'export DATAPATH=' + tools_dir  + 'RNAstructure/data_tables/\n' + tools_dir + 'RNAstructure/exe/ProbKnot ' + seqs_dir + meta + ' ' + cts_dir + meta + '.ct --sequence'
        os.system(command)


def from_knotty(fasta_file):
    seqs = list(SeqIO.parse(open(fasta_file), 'fasta'))
    out_file = data_dir + 'out_knotty.txt'
    for sample in seqs:
        meta, seq = str(sample.description), str(sample.seq)
        open(out_file, 'a').write('>' + meta + '\n')
        command = 'cd ' + tools_dir + 'Knotty\n' + './knotty ' + seq + ' >>' + out_file
        os.system(command) 


def process_parsed(in_dir, seq_file):
    codes = {'A': 32, 'C': 64, 'G': 96, 'U': 128, 'T': 128}
    def process_img(img, seq):
        for i in range(len(img)):
            for j in range(i + 1):
                if i == j:
                    img[i][i] = codes[seq[i]]
                elif img[i][j] == 255:
                    img[j][i] = 255
                    img[i][j] = 0
                    img[j + 1][i - 1] = 255
                    img[j + 2][i - 2] = 255
        return img
    
    seqs = list(SeqIO.parse(open(seq_file), 'fasta'))
    files = glob.glob(in_dir + '*.bmp')
    for file in files:
        name = file.split('/')[-1].split('.')[0]
        seq = ''
        for sample in seqs:
            meta, s = str(sample.description), str(sample.seq)
            if name == meta:
                seq = s
                break
        img = np.array(Image.open(file).convert('L'))
        img = process_img(img, seq) 
        os.remove(file)
        Image.fromarray(img).save(file.replace('.bmp', '.png'))
    
    
def from_mymodel(fasta_file):
    parsed_dir = data_dir + 'parsed/'
    mkdir(parsed_dir)
    command = 'cd /home/polina/Desktop/YaccConstructor/src/SecondaryStructureExtracter/bin/Debug/ \n./SecondaryStructureExtracter.exe -g "/home/polina/Desktop/grammar.txt" -i "' + fasta_file + '" -o "' + parsed_dir + '"'
    os.system(command) 
    
    process_parsed(parsed_dir, fasta_file)

    pred_dir = data_dir + 'pred/'
    mkdir(pred_dir)
    command = 'cd ' + data_dir + '\nconda run -n venv python3 mymodel_predict.py' 
    os.system(command) 
    
    
def prepare_fasta(in_fasta, out_fasta, size):
    seqs = list(SeqIO.parse(open(in_fasta), 'fasta')) 
    rn.shuffle(seqs)
    cnt = 0
    with open(out_fasta, 'a') as out:
        for sample in seqs:
            meta, seq = str(sample.description), str(sample.seq) 
            if len(seq) <= 100 and cnt < size:
                cnt += 1
                out.write('>' + meta + '\n' + seq + '\n')
                


def run_test(from_tool, fasta_file):
    start = time.time()
    for i in range(10):
        from_tool(fasta_file)
    end = time.time()
    print(from_tool, (end - start) / 10)
    
    