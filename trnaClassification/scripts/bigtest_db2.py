import csv
from Bio import SeqIO
import random as rn


#http://trna.ie.niigata-u.ac.jp/cgi-bin/trnadb/index.cgi database downloads procesing
#change here
path = '.../db2_data/'
ep_files = [path + 'prok.fasta', path + 'euk.fasta']
abfp_files = [path + 'arch.fasta', path + 'bact.fasta', path + 'fu.fasta', path + 'pl.fasta']
db_file = path + 'ref_db.csv

       
def align(seq, symbol='D', length=220):
    if len(seq) < length:
        return seq + symbol * (length - len(seq))
    else:
        return seq[:length]

def replace_nucls(seq):
    map_dict = {'W': ['A', 'T'], 'S': ['C', 'G'], 'M': ['A', 'C'], 'K': ['G', 'T'],
         'R': ['A', 'G'], 'Y': ['C', 'T'], 'B': ['C', 'G', 'T'], 'D': ['A', 'G', 'T'],
         'H': ['A', 'C', 'T'], 'V': ['A', 'C', 'G'], 'N': ['A', 'C', 'G', 'T'], 'Z': ['']}
    for nucl in seq:
        if nucl in map_dict.keys():
            seq = seq.replace(nucl, rn.choice(map_dict[nucl]))
    return seq

def get_ep():   #get fungi and plants from eukaryotic file
    out_file = path + 'bigtest.csv'
    cnt = 0
    with open(db_file, 'w') as db_out, open(out_file, 'w') as out:
        db = csv.writer(db_out, delimiter=',')
        db.writerow(['id', 'meta', 'class'])
        for i in range(2):
                inp = list(SeqIO.parse(open(ep_files[i]), 'fasta'))
                for sample in inp:
                    meta, seq = sample.description, str(sample.seq)
                    if not 'andidatus' in meta and not 'nclassified' in meta:
                        cls = ep_files[i].split('/')[-1].split('.')[0][0]
                        seq = align(replace_nucls(''.join(list(filter(lambda x: x == x.upper(), seq)))))
                        db.writerow([str(cnt), meta, cls])
                        out.write('>' + str(cnt) + ',' + ','.join(seq) + '\n')
                        cnt += 1

def get_abfp(): #write all archaeal, bacterial, fungi and plant samples to csv file
    out_file = path + 'bigtest.csv'
    cnt = 0
    with open(db_file, 'w') as db_out, open(out_file, 'w') as out:
        db = csv.writer(db_out, delimiter=',')
        db.writerow(['id', 'meta', 'class'])
        for i in range(4):
                inp = list(SeqIO.parse(open(abfp_files[i]), 'fasta'))
                for sample in inp:
                    meta, seq = sample.description, str(sample.seq)
                    if not 'andidatus' in meta and not 'nclassified' in meta:
                        cls = apfp_files[i].split('/')[-1].split('.')[0][0]
                        seq = align(replace_nucls(''.join(list(filter(lambda x: x == x.upper(), seq)))))
                        db.writerow([str(cnt), meta, cls])
                        out.write('>' + str(cnt) + ',' + ','.join(seq) + '\n')
                        cnt += 1


#get_ep()
#get_abfp()
