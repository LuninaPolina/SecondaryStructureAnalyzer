'''Load and transform data from different RNA secondary structure databases'''

import os
import glob
import random as rn
from PIL import Image
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
import zipfile
from Bio import SeqIO
import re

#RNAstructure tool should be installed to the tools_dir because it is used for ct2db transformation
#RNAview tool should be installed to the tools_dir because it is used for pdb data transformation
tools_dir = ''
data_dir = ''
MIN_LEN, MAX_LEN = 0, 200


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
        f2 = dbs_dir + f.split('/')[-1].replace('.ct', '.db')
        command = 'export DATAPATH=' + rnastr_dir + 'data_tables/\n' + rnastr_dir + '/exe/ct2dot ' + f + ' 0 ' + f2 
        os.system(command)
        

def plot_distr(seqs, out_file):
    distr = {}
    for seq in seqs:
        if len(seq) in distr.keys():
            distr[len(seq)] += 1
        else:
            distr[len(seq)] = 1
    plt.bar(distr.keys(), distr.values())
    plt.savefig(out_file, dpi=200)
        

def from_rnastrand(src_dir, out_fasta, out_refdb, out_distr, id_cnt):
    seqs = []
    cts_dir = src_dir + 'cts/'
    mkdir(cts_dir)
    open(out_refdb, 'w').write('id\tmeta\tdatabase\n')
    files = sorted(glob.glob(src_dir + 'all_ct_files/*.ct'))
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
            data = open(f, encoding = 'ISO-8859-1').read().split('\n')
            meta, seq, dot = data[0][1:], data[1].upper(), data[2]
            if all(s in 'ACGU' for s in seq) and not seq in seqs: 
                out.write('>' + str(id_cnt) + '\n' + seq + '\n' + dot + '\n')
                db.write(str(id_cnt) + '\t' + meta + '\t' + 'RNAstrand' + '\n')
                seqs.append(seq)
                id_cnt += 1
    os.system('rm -r ' + cts_dir)
    os.system('rm -r ' + dbs_dir)
    plot_distr(seqs, out_distr)


def from_pseudobase(in_fasta, out_fasta, out_refdb, id_cnt):
    seqs = []
    data = open(in_fasta).read().split('\n')
    with open(out_fasta, 'w') as out, open(out_refdb, 'a') as db:
        for i in range(0, len(data) - 2, 3):
            meta, seq, dot = data[i][2:], data[i + 1], data[i + 2]
            if all(s in 'ACGU' for s in seq) and not seq in seqs and not '~' in dot:
                out.write('>' + str(id_cnt) + '\n' + seq + '\n' + dot + '\n')
                db.write(str(id_cnt) + '\t' + meta + '\t' + 'Pseudobase' + '\n') 
                seqs.append(seq)
                id_cnt += 1


def from_rnacentral(in_fasta, out_fasta, ref_db, id_cnt):
    data = open(in_fasta).read().split('\n')
    with open(out_fasta, 'w') as out, open(out_refdb, 'a') as db:
        for i in range(0, len(data) - 2, 2):
            meta, seq = data[i][1:], data[i + 1]
            out.write('>' + str(id_cnt) + '\n' + seq.replace('T', 'U') + '\n')
            db.write(str(id_cnt) + '\t' + meta + '\t' + 'RNAcentral' + '\n') 
            id_cnt += 1
            

def from_gutell_lab(src_dir, out_fasta, ref_db, out_distr, id_cnt):
    dirs = glob.glob(src_dir + '*')
    seqs = []
    open(ref_db, 'w').write('id\tmeta\tdatabase\n')
    for d in dirs:
        cts_dir = data_dir + 'gutell_lab/cts/'
        mkdir(cts_dir)
        files = glob.glob(d + '/*.ct')
        for f in files:
            f2 = cts_dir + f.split('/')[-1]
            with open(f, 'r') as inp, open(f2, 'w') as out:
                ct = inp.read().split('\n')
                seq_info = d.split('/')[-1] + ' ' + ' '.join(ct[0:5])
                seq_len = ct[4].split(' ')[0]
                new_ct = seq_len + ' ' + seq_info + '\n' + '\n'.join(ct[5:])
                out.write(new_ct)
        dbs_dir = data_dir + 'gutell_lab/dbs/'
        mkdir(dbs_dir)
        cts2dbs(cts_dir, dbs_dir)
        files = sorted(glob.glob(dbs_dir + '*.db'))
        with open(out_fasta, 'a') as out, open(out_refdb, 'a') as db:
            for f in files:
                data = open(f).read().split('\n')
                meta, seq, dot = data[0][1:], data[1].upper(), data[2]
                seq = seq.replace('T', 'U')
                if all(s in 'ACGU' for s in seq) and not seq in seqs:
                    out.write('>' + str(id_cnt) + '\n' + seq + '\n' + dot + '\n')
                    db.write(str(id_cnt) + '\t' + meta + '\t' + 'Gutell Lab CRW' + '\n')
                    seqs.append(seq)
                    id_cnt += 1
        os.system('rm -r ' + cts_dir)
        os.system('rm -r ' + dbs_dir)
    plot_distr(seqs, out_distr)
        

def from_tmrna(src_dir, out_fasta, ref_db, out_distr, id_cnt):
    seqs = []
    files = glob.glob(src_dir + '*.ct')
    dbs_dir = data_dir + 'tmRNA/dbs/'
    mkdir(dbs_dir)
    cts2dbs(src_dir, dbs_dir)
    files = sorted(glob.glob(dbs_dir + '*.db'))
    with open(out_fasta, 'w') as out, open(out_refdb, 'w') as db:
        db.write('id\tmeta\tdatabase\n')
        for f in files:
            data = open(f).read().split('\n')
            meta, seq, dot = data[0][1:], data[1].upper(), data[2]
            seq = seq.replace('T', 'U')
            if all(s in 'ACGU' for s in seq) and not seq in seqs: 
                out.write('>' + str(id_cnt) + '\n' + seq + '\n' + dot + '\n')
                db.write(str(id_cnt) + '\t' + meta + '\t' + 'tmRNA Database' + '\n')
                seqs.append(seq)
                id_cnt += 1
        os.system('rm -r ' + dbs_dir)
    plot_distr(seqs, out_distr)
    

def from_srp(src_dir, out_fasta, ref_db, out_distr, id_cnt):
    seqs = []
    files = glob.glob(src_dir + '*.ct')
    dbs_dir = data_dir + 'SRP/dbs/'
    mkdir(dbs_dir)
    cts2dbs(src_dir, dbs_dir)
    files = sorted(glob.glob(dbs_dir + '*.db'))
    with open(out_fasta, 'w') as out, open(out_refdb, 'w') as db:
        db.write('id\tmeta\tdatabase\n')
        for f in files:
            data = open(f).read().split('\n')
            meta, seq, dot = data[0][1:], data[1].upper(), data[2]
            seq = seq.replace('T', 'U')
            if all(s in 'ACGU' for s in seq) and not seq in seqs: 
                out.write('>' + str(id_cnt) + '\n' + seq + '\n' + dot + '\n')
                db.write(str(id_cnt) + '\t' + meta + '\t' + 'SRP Database' + '\n')
                seqs.append(seq)
                id_cnt += 1
        os.system('rm -r ' + dbs_dir)
    plot_distr(seqs, out_distr)
    

def from_sprinzl(src_dir, out_fasta, ref_db, out_distr, id_cnt):
    seqs = []
    files = glob.glob(src_dir + '*.fst')
    with open(out_fasta, 'w') as out, open(out_refdb, 'w') as db:
        db.write('id\tmeta\tdatabase\n')
        for f in files:
            data = open(f).read().split('\n')
            for i in range(0, len(data) - 2, 3):
                meta, seq, dot = data[i][1:], data[i + 1], data[i + 2] 
                seq = seq.replace('T', 'U')
                if all(s in 'ACGU' for s in seq) and not seq in seqs: 
                    out.write('>' + str(id_cnt) + '\n' + seq + '\n' + dot + '\n')
                    db.write(str(id_cnt) + '\t' + meta + '\t' + 'Sprinzl tRNA Database' + '\n')
                    seqs.append(seq)
                    id_cnt += 1
    plot_distr(seqs, out_distr)
    

def from_rfam(src_file, out_fasta, ref_db, out_distr, id_cnt):
    seqs = []
    with open(out_fasta, 'w') as out, open(out_refdb, 'w') as db:
        db.write('id\tmeta\tdatabase\n')
        data = open(src_file, encoding = 'ISO-8859-1').read().split('\n')
        for line in data:
            if len(line) > 0:
                if line == '# STOCKHOLM 1.0':
                    fam_info = ''
                    fam_seqs = []
                    fam_metas = []
                elif line == '//':
                    for i in range(len(fam_seqs)):
                        meta, seq = fam_metas[i], fam_seqs[i]
                        if all(s in 'ACGU' for s in seq) and not seq in seqs and not '~' in dot and '#=GF SS   Published' in fam_info: 
                            out.write('>' + str(id_cnt) + '\n' + seq + '\n' + dot + '\n')
                            db.write(str(id_cnt) + '\tFamily: ' + fam_info + 'Sequence: ' + meta + '\t' + 'Rfam Database' + '\n')
                            seqs.append(seq)
                            id_cnt += 1                   
                elif line[0] == '#':
                    if '#=GC SS_cons' in line:
                        dot = line.split(' ')[-1]
                        dot = dot.replace('_', '.').replace('-', '.').replace(',', '.').replace(':', '.')
                        dot = dot.replace('<', '(').replace('>', ')').replace('[', '(').replace(']', ')').replace('{', '(').replace('}', ')')
                    elif '#=GF' in line:
                        fam_info += line + ' '
                else:
                    meta, seq = line.split(' ')[0], line.split(' ')[-1]
                    seq = seq.replace('T', 'U')
                    fam_seqs.append(seq)
                    fam_metas.append(meta)
        plot_distr(seqs, out_distr)
        

def from_pdb(src_dir, src_fasta, out_fasta, out_dir, ref_db, out_log, out_distr, id_cnt):
    
    def get_bp_list(seq_id, ch_id, seq):
        manual, rm_idx, add_idx = '', [], []
        #run rnaview
        rv_dir = tools_dir + 'RNAVIEW/'
        f = src_dir + 'pdb' + seq_id.lower() + '.ent'    
        command = 'cd ' + rv_dir + '\nexport RNAVIEW=' + rv_dir + '\nexport PATH=' + rv_dir + '/bin\nrnaview ' + f
        os.system(command)
        try:
            f2 = f.replace('.ent', '.ent.out')
            data = open(f2).read()
        except:
            f2 = f.replace('.ent', '.ent_nmr.pdb.out')
            data = open(f2).read()
        #skip samples with uncommon residues
        if len(re.findall('uncommon residue.*on chain ' + ch_id, data)) > 0:
            return [], manual, seq
        #get base pair list
        bp_list = []
        unmodeled = sorted(list(map(lambda x: int(x.split(' ')[-1]), re.findall('REMARK 465.*' + ch_id + ' +[0-9]+', open(f).read()))))
        bp_str = re.search('BEGIN_base-pair(.|\n)*END_base-pair', data).group(0).split('\n')[1:-1]
        for line in bp_str:
            pair = re.search('[0-9]+_[0-9]+', line).group(0).split('_')
            pair_chains = list(map(lambda x: x.replace(' ', '').replace(':', '') ,re.findall(' [a-zA-Z0-9]:', line)))
            if pair_chains[0] == ch_id or pair_chains[1] == ch_id:
                #skip intersecting chains
                if pair_chains[0] != pair_chains[1]:
                    return [], manual, seq
                #if some problems with indexation then process manually
                if int(pair[0]) > len(seq) - len(unmodeled) or int(pair[1]) > len(seq) - len(unmodeled): 
                    manual = ' oom idx'
                    #open(out_log, 'a').write(seq_id + ' ' + ch_id + ' oom idx\n')
                    #return [], manual, seq #change here!!
                if not 'stacked' in line:
                    bp_list.append('_'.join(pair))
        #get unmodeled residues numbers
        if len(bp_list) > 0:
            #get first residue numeration
            try:
                start = int(re.search('chain_ID: +' + ch_id + '.*\n', open(f.replace('.ent', '.ent_log.out')).read()).group(0).split('residue')[-1].split('to')[0].strip())
                end = int(re.search('chain_ID: +' + ch_id + '.*\n', open(f.replace('.ent', '.ent_log.out')).read()).group(0).split('to')[-1].strip())
            except:
                start = int(re.search('chain_ID: +' + ch_id + '.*\n', open(f.replace('.ent', '.ent_nmr.pdb_log.out')).read()).group(0).split('residue')[-1].split('to')[0].strip())
                end = int(re.search('chain_ID: +' + ch_id + '.*\n', open(f.replace('.ent', '.ent_nmr.pdb_log.out')).read()).group(0).split('to')[-1].strip())
            if len(unmodeled) > 0:
                start = min(start , unmodeled[0]) 
                end = max(end , unmodeled[-1]) 
            if end - start + 1 != len(seq):
                #open(out_log, 'a').write(seq_id + ' ' + ch_id + ' auth idx\n')
                #return [], manual, seq
                manual += ' auth idx'
            if len(unmodeled) > 0:
                #numeration from 1
                unmodeled = list(map(lambda x: x - start + 1, unmodeled))
                #if all unmodeled residues are placed at the edges then cut them else process this sample manually
                seq2 = seq
                for r in unmodeled: 
                    seq2 = seq2[:r - 1] + ' ' + seq2[r:]
                seq2 = seq2.strip()
                if ' ' in seq2:
                    rm_idx = []
                    for i in range(len(unmodeled)):
                        if unmodeled[i] == i + 1:
                            rm_idx.append(unmodeled[i])
                        else:
                            break
                    for i in range(len(unmodeled)):
                        if unmodeled[-1 - i] == len(seq) - i:
                            rm_idx.append(unmodeled[-1 - i])
                        else:
                            break
                    add_idx = list(filter(lambda x: not x in rm_idx, unmodeled))
                    if len(add_idx) / len(seq) > 0.1: #change here!
                        #open(out_log, 'a').write(seq_id + ' ' + ch_id + ' ' + str(len(add_idx)) + '/' + str(len(seq2)) + ' inner unmodeled\n')
                        #return [], manual, seq
                        manual += ' ' + str(len(add_idx)) + '/' + str(len(seq2)) + ' inner unmodeled'
                else:
                    seq = seq2
            #edit base pairs and seq for samples with unmodeled residues
            if len(rm_idx) + len(add_idx) > 0:
                bp_list, seq = process_unmodeled(bp_list, seq, add_idx, rm_idx)
        return bp_list, manual, seq
                   
    def get_independent(seq_id, chains):
        independent_chains = dict()
        for ch in chains.keys():
            bp_list = get_bp_list(seq_id, ch, chains[ch])
            if len(bp_list[0]) > 0:
                independent_chains[ch] = bp_list
        return independent_chains
    
    def process_unmodeled(bp_list, seq, add_idx, rm_idx):
        #add inner unmodeled to all base pairs
        for k in range(len(bp_list)):
            n1, n2 = int(bp_list[k].split('_')[0]), int(bp_list[k].split('_')[1])
            for i in add_idx:
                if n1 >= i:
                    n1 += 1
                if n2 >= i:
                    n2 += 1
            bp_list[k] = str(n1) + '_' + str(n2)
        #remove outer unmodeled from seq
        for i in rm_idx:
            seq = seq[:i - 1] + ' ' + seq[i:]
        seq = seq.replace(' ', '')
        return bp_list, seq
                                  
    global id_cnt
    files = glob.glob(src_dir + '*.*')
    for f in files:
        if not f.endswith('.ent'):
            os.system('rm ' + f)
    seqs = [] 
    #seqs = list(map(lambda x: str(x.seq), list(SeqIO.parse(open('existing_seqs.fasta'), 'fasta'))))
    src_seqs = list(SeqIO.parse(open(src_fasta), 'fasta'))
    chains = {}
    prev_id = ''
    mkdir(out_dir)
    time_cnt = 0
    flag = False
    with open(out_fasta, 'a') as out, open(out_refdb, 'a') as db:
        db.write('id\tmeta\tdatabase\n') 
        for sample in src_seqs:
            time_cnt += 1
            if time_cnt % 10000 == 0:
                print(time_cnt, 'done')
            if True:
                f = src_dir + 'pdb' + id.lower() + '.ent'
                if f in files:
                    if prev_id != id and len(prev_id) > 0:
                        if len(chains) == 1: #for one-chain entries
                            if (prev_id, list(chains.keys())[0]) in samples_oom:
                                bp_list, manual, seq = get_bp_list(prev_id, list(chains.keys())[0], seq)
                                if all(s in 'ACGU' for s in seq) and not seq in seqs and len(bp_list) > 0: 
                                    if len(manual) > 0:
                                        print(prev_id, manual)
                                        print(bp_list)
                                    if 'oom idx' in manual or 'auth idx' in manual:
                                        shift = int(input('shift'))
                                        for i in range(len(bp_list)):
                                            n1, n2 = int(bp_list[i].split('_')[0]), int(bp_list[i].split('_')[1])
                                            n1 -= shift
                                            n2 -= shift
                                            bp_list[i] = str(n1) + '_' + str(n2) 
                                    if len(manual) == 0 or input('y/n') == 'y':
                                        out.write('>' + str(id_cnt) + '\n' + seq + '\n')
                                        db.write(str(id_cnt) + '\t' + meta + '\t' + 'PDB' + '\n')
                                        open(out_dir + str(id_cnt) + '.txt', 'w').write('\n'.join(bp_list))
                                        seqs.append(seq)
                                        id_cnt += 1  
                        else: #for multi-chain entries
                            if prev_id in list(map(lambda x: x[0], samples_oom)):
                                independent_chains = get_independent(prev_id, chains)
                                for ch in independent_chains.keys():
                                    if (prev_id, ch) in samples_oom:
                                        bp_list, manual, seq = independent_chains[ch]  
                                        if all(s in 'ACGU' for s in seq) and not seq in seqs and len(bp_list) > 0: 
                                            if len(manual) > 0:
                                                print(prev_id, 'chain', ch, manual)
                                                print(bp_list)
                                            if 'oom idx' in manual or 'auth idx' in manual:
                                                shift = int(input('shift'))
                                                for i in range(len(bp_list)):
                                                    n1, n2 = int(bp_list[i].split('_')[0]), int(bp_list[i].split('_')[1])
                                                    n1 -= shift
                                                    n2 -= shift
                                                    bp_list[i] = str(n1) + '_' + str(n2) 
                                            if len(manual) == 0 or input('y/n') == 'y':
                                                out.write('>' + str(id_cnt) + '\n' + seq + '\n')
                                                db.write(str(id_cnt) + '\t' + meta + ' (Chain ' + ch + ')\t' + 'PDB' + '\n')
                                                open(out_dir + str(id_cnt) + '.txt', 'w').write('\n'.join(bp_list))
                                                seqs.append(seq)
                                                id_cnt += 1  
                        chains = {}
                    meta, seq = str(sample.description), str(sample.seq).upper().replace('T', 'U')
                    chains_info = re.sub('[A-Z]\[auth ', '', meta.split('|')[1]).replace(']', '').replace(',', '').split(' ')[1:]
                    for el in chains_info:
                        chains[el] = seq
                    prev_id = id
    plot_distr(seqs, out_distr)
    files = glob.glob(src_dir + '*.*')
    for f in files:
        if not f.endswith('.ent'):
            os.system('rm ' + f)
            
            
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

