'''Load, transform and process data from PDB database'''

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
from shutil import copyfile


#RNAview tool should be installed to the tools_dir because it is used for pdb data transformation
tools_dir = ''


def get_bp_list_check(seq_id, ch_id, seq):
    problem  = ''
    rv_dir = ''
    f = src_dir + 'pdb' + seq_id.lower() + '.ent'    
    command = 'cd ' + rv_dir + '\nexport RNAVIEW=' + rv_dir + '\nexport PATH=' + rv_dir + '/bin\nrnaview ' + f
    os.system(command)
    try:
        f2 = f.replace('.ent', '.ent.out')
        data = open(f2).read()
    except:
        f2 = f.replace('.ent', '.ent_nmr.pdb.out')
        data = open(f2).read()
    if len(re.findall('uncommon residue.*on chain ' + ch_id, data)) > 0:
        return [], problem
    nucl_list = []
    unmodeled = sorted(list(map(lambda x: int(x.split(' ')[-1]), re.findall('REMARK 465.*' + ch_id + ' +[0-9]+', open(f).read()))))
    unmodeled += sorted(list(set(list(map(lambda x: int(x.split(' ')[-1]), re.findall('REMARK 470.*' + ch_id + ' +[0-9]+', open(f).read()))))))
    if len(unmodeled) > 0:
        problem = 'unmodeled'
    bp_str = re.search('BEGIN_base-pair(.|\n)*END_base-pair', data).group(0).split('\n')[1:-1]
    for line in bp_str:
        pair = re.search('[0-9]+_[0-9]+', line).group(0).split('_')
        pair_chains = list(map(lambda x: x.replace(' ', '').replace(':', '') ,re.findall(' [a-zA-Z0-9]:', line)))
        if pair_chains[0] == ch_id or pair_chains[1] == ch_id:
            nucls = re.search('[ACGU]-[ACGU]', line).group(0)
            if pair_chains[0] != pair_chains[1]:
                return [], problem
            if int(pair[0]) > len(seq) - len(unmodeled) or int(pair[1]) > len(seq) - len(unmodeled): 
                if not 'oom idx' in problem:
                    problem += ' oom idx'
            if not 'stacked' in line:
                nucl_list.append(nucls)
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
        problem += ' auth idx'
    return nucl_list, problem


def get_bp_list(seq_id, ch_id, seq):
    manual, rm_idx, add_idx = '', [], []
    #run rnaview
    f = src_dir + 'pdb' + seq_id.lower() + '.ent'    
    command = 'cd ' + tools_dir + '\nexport RNAVIEW=' + tools_dir + '\nexport PATH=' + rv_dir + '/bin\nrnaview ' + f
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
    part_modeled = sorted(list(set(list(map(lambda x: int(x.split(' ')[-1]), re.findall('REMARK 470.*' + ch_id + ' +[0-9]+', open(f).read()))))))
    print(unmodeled)
    print(part_modeled)
    if len(part_modeled) > 0:
        print(seq_id, ch_id, 'partially modeled ' + ', '.join(list(map(lambda x: str(x), part_modeled))))
        if input('treat as unmodeled?') == 'y':
            manual = ' partially modeled'
            unmodeled = sorted(unmodeled + part_modeled)
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
                if not 'oom idx' in manual:
                    manual += ' oom idx'
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
                if True: #len(add_idx) / len(seq) > 0.2: 
                    #open(out_log, 'a').write(seq_id + ' ' + ch_id + ' ' + str(len(add_idx)) + '/' + str(len(seq2)) + ' inner unmodeled\n')
                    #return [], manual, seq
                    manual += ' ' + str(len(add_idx)) + '/' + str(len(seq2)) + ' inner unmodeled'
            else:
                seq = seq2
        #edit base pairs and seq for samples with unmodeled residues
        if len(rm_idx) + len(add_idx) > 0:
            bp_list, seq = process_unmodeled(bp_list, seq, add_idx, rm_idx)
    print('\n'.join(bp_list), seq)
    return bp_list, manual, seq


def get_independent(seq_id, chains):
    independent_chains = dict()
    for ch in chains.keys():
        bp_list = get_bp_list(seq_id, ch, chains[ch])
        if len(bp_list[0]) > 0:
            independent_chains[ch] = bp_list
    return independent_chains


def process_unmodeled(bp_list, seq, add_idx, rm_idx):
    print(add_idx, rm_idx)
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
     
    
def from_pdb(src_dir, src_fasta, out_fasta, out_dir, ref_db, out_log, out_distr, id_cnt):
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
    with open(out_fasta, 'a') as out, open(out_refdb, 'a') as db:
        db.write('id\tmeta\tdatabase\n') 
        for sample in src_seqs:
            time_cnt += 1
            if time_cnt % 10000 == 0:
                print(time_cnt, 'done')
            id = str(sample.description).split('_')[0]
            f = src_dir + 'pdb' + id.lower() + '.ent'
            if f in files:
                if prev_id != id and len(prev_id) > 0:
                    if len(chains) == 1: #for one-chain entries
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
                        independent_chains = get_independent(prev_id, chains)
                        for ch in independent_chains.keys():
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
 

def from_pdb_pm(src_dir, src_fasta, out_fasta, out_dir, ref_db):
    src_seqs = list(SeqIO.parse(open(src_fasta), 'fasta'))
    seqs = list(SeqIO.parse(open(out_fasta), 'fasta'))
    time_cnt = 0
    db = pd.read_csv(open(ref_db), sep='\t')
    for sample in seqs:
        id, seq = str(sample.description), str(sample.seq)
        meta = db.loc[db['id'] == int(id)].values[0][1]
        src_id = meta.split('_')[0]
        chain = re.search('\(Chain.+\)', meta)
        if chain is not None:
            chain = chain.group(0).split(' ')[-1][:-1]
        else:
            chain = meta.split('|')[1].split(' ')[-1]
        f = src_dir + 'pdb' + src_id.lower() + '.ent'
        if f in files:
            src_seq = ''
            for src_sample in src_seqs:
                if str(src_sample.description) == meta.split(' (Chain')[0]:
                    src_seq = str(src_sample.seq).upper().replace('T', 'U')
            if len(src_seq) == 0:
                print(src_id, id)
            bp_list, manual, seq = get_bp_list(src_id, chain, src_seq)
            if 'partially' in manual:
                print(src_id, 'chain', chain, manual)
                print(bp_list)
                if 'oom idx' in manual or 'auth idx' in manual:
                    shift = int(input('shift'))
                    for i in range(len(bp_list)):
                        n1, n2 = int(bp_list[i].split('_')[0]), int(bp_list[i].split('_')[1])
                        n1 -= shift
                        n2 -= shift
                        bp_list[i] = str(n1) + '_' + str(n2) 
                if input('y/n') == 'y':
                    print(id, seq, len(seq))
                    open(out_dir + str(id) + '.txt', 'w').write('\n'.join(bp_list))
            time_cnt += 1
            if time_cnt % 100 == 0:
                print(time_cnt, 'done')          


def from_pdb_check(src_dir, src_fasta, out_fasta, out_dir, ref_db):
    src_seqs = list(SeqIO.parse(open(src_fasta), 'fasta'))
    seqs = list(SeqIO.parse(open(out_fasta), 'fasta'))
    time_cnt = 0
    db = pd.read_csv(open(ref_db), sep='\t')
    files = glob.glob(src_dir + '*.*')
    flag = False
    for sample in seqs:
        id, seq = str(sample.description), str(sample.seq)
        if id == '34959':
            flag = True
        if flag:
            meta = db.loc[db['id'] == int(id)].values[0][1]
            src_id = meta.split('_')[0]
            chain = re.search('\(Chain.+\)', meta)
            if chain is not None:
                chain = chain.group(0).split(' ')[-1][:-1]
            else:
                chain = meta.split('|')[1].split(' ')[-1].replace(']', '')
            f = src_dir + 'pdb' + src_id.lower() + '.ent'
            if f in files:
                src_seq = ''
                for src_sample in src_seqs:
                    if str(src_sample.description) == meta.split(' (Chain')[0]:
                        src_seq = str(src_sample.seq).upper().replace('T', 'U')
                if len(src_seq) == 0:
                    print(src_id, id)
                nucl_list, problem = get_bp_list_check(src_id, chain, src_seq)
                bp_list = open(out_dir + str(id) + '.txt').read().split('\n')
                info = id + ', ' + src_id + ', ' + chain + ': '
                for line in bp_list:
                    if len(line) > 0:
                        n1, n2 = int(line.split('_')[0]), int(line.split('_')[1])
                        if not str(seq[n1 - 1]) + '-' + str(seq[n2 - 1]) in nucl_list:
                            info += '\nno base pair ' + line
                        else:
                            nucl_list.remove(str(seq[n1 - 1]) + '-' + str(seq[n2 - 1]))
                if len(nucl_list) > 0:
                    info += '\nextra base pairs ' + ' '.join(nucl_list)
                if not 'base pair' in info and len(problem) == 0:
                    info += 'no problem'
                else:
                    info += problem 
                    if 'unmodeled' in problem:
                        info += '\n' + src_seq + ' ' + str(len(src_seq)) + '\n' + seq + ' ' + str(len(seq)) + '\n' 
                    elif src_seq != seq:
                        info += '\nseq trouble\n' + src_seq + ' ' + str(len(src_seq)) + '\n' + seq + ' ' + str(len(seq)) + '\n' 
                print(info)
                if 'no problem' in info or input('y/n') == 'y':
                    continue


def clean(src_dir):    
    files = glob.glob(src_dir + '*.*')
    for f in files:
        if not f.endswith('.ent'):
            os.system('rm ' + f)   
 

def plot_distr(seqs, out_file):
    distr = {}
    for seq in seqs:
        if len(seq) in distr.keys():
            distr[len(seq)] += 1
        else:
            distr[len(seq)] = 1
    plt.bar(distr.keys(), distr.values())
    plt.savefig(out_file, dpi=200)


def get_substructures(in_dir, in_fasta, out_dir, max_len, min_len, window):
    
    def make_slice(img, start, end, check_none=True):
        n = end - start
        slice = np.array(Image.new('L', (n, n), (0))) 
        for i0 in range(start, end):
            for j0 in range(start, end):
                slice[i0 - start][j0 - start] = img[i0][j0]
                if check_none:
                    for i in range(start):
                        if img[i][i0] == 255 or img[i][j0] == 255 or img[i0][i] == 255 or img[j0][i] == 255:
                            return None   
                    for i in range(end, len(img)):
                        if img[i][i0] == 255 or img[i][j0] == 255 or img[i0][i] == 255 or img[j0][i] == 255:
                            return None                          
        return slice
    
    def get_seq(img):
        codes = {32: 'A', 64: 'C', 96: 'G', 128: 'U'}
        seq = ''
        for i in range(len(img)):
            seq += codes[img[i][i]]
        return seq
    
    files = glob.glob(in_dir + '*.png')
    seqs = list(map(lambda x: str(x.seq), list(SeqIO.parse(open(in_fasta), 'fasta'))))
    for f_out in files:
        f_in = f_out.replace('/out/' , '/in/')
        img_out = np.array(Image.open(f_out))
        img_in = np.array(Image.open(f_in))
        if len(img_out) <= 200:
            copyfile(f_out, f_out.replace(in_dir, out_dir))
            copyfile(f_in, f_out.replace(in_dir, out_dir).replace('/out/', '/in/'))
        else:
            print('process image', f_out.split('/')[-1], 'with length', len(img_out))
            for n in range(max_len, min_len, -1 * window):
                for start in range(len(img_out) - n):
                    end = start + n
                    slice_out = make_slice(img_out, start, end, True)
                    if slice_out is not None:
                        slice_in = make_slice(img_in, start, end, False)
                        seq = get_seq(slice_in)
                        if not seq in seqs:
                            Image.fromarray(slice_out).save(f_out.replace(in_dir, out_dir).replace('.png', '_' + str(start) + '_' + str(end) + '.png'))
                            Image.fromarray(slice_in).save(f_out.replace(in_dir, out_dir).replace('/out/', '/in/').replace('.png', '_' + str(start) + '_' + str(end) + '.png'))
                            seqs.append(seq)
            