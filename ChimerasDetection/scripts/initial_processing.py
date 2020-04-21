from Bio import SeqIO
import csv
import random
from random import randint
import re
import os

def replace_nucls(seq):
    map_dict = {'W': ['A', 'T'], 'S': ['C', 'G'], 'M': ['A', 'C'], 'K': ['G', 'T'],
         'R': ['A', 'G'], 'Y': ['C', 'T'], 'B': ['C', 'G', 'T'], 'D': ['A', 'G', 'T'],
         'H': ['A', 'C', 'T'], 'V': ['A', 'C', 'G'], 'N': ['A', 'C', 'G', 'T'], 'Z': ['']}
    for nucl in seq:
        if nucl in map_dict.keys():
            seq = seq.replace(nucl, random.choice(map_dict[nucl]))
    return seq

def filter_cond(seq):
    return (len(seq) <= 1024)

def align(seq, symbol, length):
    return seq + symbol * (length - len(seq))
    
def process_sequences(args):
    src, out_file_16s, out_file_chim, info_file = args
    with open(info_file, 'w') as csv_out, open(out_file_16s, 'w') as out_16s, open(out_file_chim, 'w') as out_chim:
        csv_writer = csv.writer(csv_out, delimiter=',')
        seq_id, cnt_16s, cnt_chim = 0, 0, 0
        fst_line_16s, fst_line_chim = True, True
        for file in src:
            inp = SeqIO.parse(open(file), 'fasta')         
            for sample in inp:
                meta, seq = sample.description, str(sample.seq)
                if filter_cond(seq):
                    cls = 'p' if '16s' in file else 'n'
                    seq = align(replace_nucls(seq),'D',1024)
                    if cls == 'p':
                        if fst_line_16s:
                            out_16s.write('>' + str(seq_id) + '\n' + seq)
                            fst_line_16s = False                    
                        else:
                            out_16s.write('\n>' + str(seq_id) + '\n' + seq)
                        cnt_16s += 1
                    else:
                        if fst_line_chim:
                            out_chim.write('>' + str(seq_id) + '\n' + seq)
                            fst_line_chim = False
                        else:
                            out_chim.write('\n>' + str(seq_id) + '\n' + seq)
                        cnt_chim += 1
                        meta += '(chimeric)'   
                    csv_writer.writerow([str(seq_id), meta, cls])
                    seq_id += 1
        print('chimeric: ' + str(cnt_chim) + '\n16s: ' + str(cnt_16s))

def process_genomes(args):
    src, out_file = args
    with open(out_file, 'w') as out:
        fst_line = True
        for path, subdirs, files in os.walk(src):        
            for name in files:
                file = os.path.join(path, name)
                inp = SeqIO.parse(open(file), 'fasta')
                for sample in inp:
                    meta, seq = sample.description, str(sample.seq)
                    regions = ['0'] + re.split(':| ', meta.split(',')[-1])[1:] + [str(len(seq))]
                    for i in range(0, len(regions) - 1, 2):
                        fr, to = regions[i], regions[i + 1]
                        new_meta = ';'.join(meta.split(',')[:-1]) + '; ' + fr + ':' + to + '(genome)'
                        new_seq = replace_nucls(seq[int(fr):int(to)])
                        if fst_line:
                            out.write('>' + new_meta + '\n' + new_seq)
                            fst_line = False
                        else:
                            out.write('\n>' + new_meta + '\n' + new_seq)

def generate_genomic(args):
    inp_file, out_file, info_file = args
    inp = SeqIO.parse(open(inp_file), 'fasta')
    with open(info_file, 'w') as csv_out, open(out_file,'w') as out:
        csv_writer = csv.writer(csv_out, delimiter=',')
        seq_id = 971900
        fst_line = True
        for sample in inp:
            meta, seq = sample.description, str(sample.seq)
            num_samples = int(len(seq) / 1024)
            positions = [randint(0, len(seq) - 1024) for i in range(num_samples)]
            for pos in positions:
                length = randint(200, 1024)
                new_meta = meta + '; ' + str(pos) + '+' + str(length)
                csv_writer.writerow([str(seq_id), new_meta, 'n'])
                if pos + length < len(seq):
                    new_seq = seq[pos:pos + length]
                else:
                    new_seq = seq[pos:]
                new_seq = new_seq + 'D' * (1024 - len(new_seq))
                if fst_line:
                    out.write('>' + str(seq_id) + '\n' + new_seq)
                    fst_line = False
                else:
                    out.write('\n>' + str(seq_id) + '\n' + new_seq)
                seq_id += 1
        print('genomic: ' + str(seq_id - 971900))

def process_all(args_seq, args_genomic):
    process_sequences(args_seq)
    process_genomes(args_genomic[:2])
    generate_genomic(args_genomic[1:])