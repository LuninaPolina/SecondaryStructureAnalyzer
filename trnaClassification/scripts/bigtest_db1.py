import csv
from Bio import SeqIO
import random as rn


#http://gtrnadb2009.ucsc.edu/ database downloads procesing
#change here
path = '.../db1_data/'
ep_files = [path + 'prok.fasta', path + 'euk.fasta']
abfp_files = [path + 'arch.fasta', path + 'bact.fasta', path + 'fu.fasta', path + 'pl.fasta']
db_file = path + 'ref_db.csv'

p_names = ['Brachypodium_distachyon', 'Physcomitrella_patens', 'Sorghum_bicolor', 'Vitis_vinifera', 'Zea_mays',
           'Arabidopsis_thaliana', 'Glycine_max', 'Medicago_truncatula', 'Oryza_sativa', 'Populus_trichocarpa']
f_names = ['Aspergillus_fumigatus', 'Candida_glabrata', 'Cryptococcus_neoformans', 'Debaryomyces_hansenii',
           'Encephalitozoon_cuniculi', 'Eremothecium_gossypii', 'Kluyveromyces_lactis', 'Magnaporthe_oryzae',
           'Saccharomyces_cerevisiae', 'Saccharomyces_cerevisiae', 'Schizosaccharomyces_pombe', 'Yarrowia_lipolytica']
       

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

def get_ep():   #write all eukaryotic and prokarytic samples to csv file
    out_file = path + 'bigtest.csv'
    cnt = 0
    with open(db_file, 'w') as db_out, open(out_file, 'w') as out:
        db = csv.writer(db_out, delimiter=',')
        db.writerow(['id', 'meta', 'class'])
        for i in range(2):
                inp = list(SeqIO.parse(open(ep_files[i]), 'fasta'))
                for sample in inp:
                    meta, seq = sample.description, align(replace_nucls(str(sample.seq)))
                    if not 'andidatus' in meta and not 'nclassified' in meta:
                        cls = ep_files[i].split('/')[-1].split('.')[0][0]
                        db.writerow([str(cnt), meta, cls])
                        out.write('>' + str(cnt) + ',' + ','.join(seq) + '\n')
                        cnt += 1

def split_euk():    #get fungi and plants from eukaryotic file
    inp = list(SeqIO.parse(open(ep_files[1]), 'fasta'))
    fungi_file, plant_file = abfp_files[2], abfp_files[3]
    with open(fungi_file, 'w') as out_f, open(plant_file, 'w') as out_p:
        for sample in inp:
             meta, seq = sample.description, str(sample.seq)
             if any(x in meta for x in p_names):
                 out_p.write('>' + meta + '\n' + seq + '\n')
             if any(x in meta for x in f_names):
                 out_f.write('>' + meta + '\n' + seq + '\n')

def get_abfp(): #write all archaeal, bacterial, fungi and plant samples to csv file
    split_euk()
    out_file = path + 'bigtest.csv'
    cnt = 0
    with open(db_file, 'w') as db_out, open(out_file, 'w') as out:
        db = csv.writer(db_out, delimiter=',')
        db.writerow(['id', 'meta', 'class'])
        for i in range(4):
                inp = list(SeqIO.parse(open(apfp_files[i]), 'fasta'))
                for sample in inp:
                    meta, seq = sample.description, align(replace_nucls(str(sample.seq)))
                    if not 'andidatus' in meta and not 'nclassified' in meta:
                        cls = abfp_files[i].split('/')[-1].split('.')[0][0]
                        db.writerow([str(cnt), meta, cls])
                        out.write('>' + str(cnt) + ',' + ','.join(seq) + '\n')
                        cnt += 1


#get_ep()
#get_abfp()
