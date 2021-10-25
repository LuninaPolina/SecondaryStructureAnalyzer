'''Set functions arguments here'''

import initial_processing
import select_for_parsing
import csv_processing
import select_for_nn1

src_seq = ['../data/source/rdp_data/16s.fasta',                     #source files with sequences                                      
                    '../data/source/rdp_data/chimeras.fasta',       
                    '../data/source/silva_data/16s.fasta',
                    '../data/source/silva_data/chimeras.fasta']
all_16s = '../data/all/16s.fasta'                                   #files with formatted and aligned to 1024 sequences
all_chim = '../data/all/chimeras.fasta'
info_16s_and_chim = '../data/all/info_16s_and_chim.csv'             #map id,meta,class for sequences

src_gen = '../data/source/genome_data'                              #source dir with complete genomes
all_gen = '../data/all/genomic.fasta'                               #file with all compete genomes
all_gen_gnr = '../data/all/genomic_generated.fasta'                 #subsequences from genomes without 16s RNA 
info_gen_gnr = '../data/all/info_genomic_generated.csv'             #map id,meta,class for genomic sequences

for_pars_16s = '../data/for_parsing/16s.fasta'                      #files with sequences selected for parsing
for_pars_chim = '../data/for_parsing/chimeras.fasta'
for_pars_gen = '../data/for_parsing/genomic_generated.fasta'

all_shuffled = '../data/for_nn1/all_shuffled.csv'                   #file with parsed sequences selected for nn1 train and valid
for_nn1_train = '../data/for_nn1/train.csv'                         
for_nn1_valid = '../data/for_nn1/valid.csv'
                    
ref_db = '../data/ref_db.csv'                                       #refernce db with columns id,meta,class,nn1 (nn1=train/valid)


'''
1) Get fasta files in appropriate format from 16s RNA and chimeric sequences. Align to 1024.
   Load all genomic subsequences excluding 16s RNA from genomes and use them to generate sequences of length 1024. 
   Change sequences metadata to id and save id, meta and sequence class (p if 16s else n) in csv files.
'''
def get_data():
    initial_processing.process_all([src_seq, all_16s, all_chim, info_16s_and_chim],
                                   [src_gen, all_gen, all_gen_gnr, info_gen_gnr])

'''
2) Get random selection for parsing from 16s, chimeras and genomic sequences.
   For small files set corresponding selection numbers and choose 'select_from_small_file' function in select_for_parsing.py.
   For big files -- set total number of sequences, selection number and choose 'select_from_small_file'.
'''
def for_parsing():
    select_for_parsing.select_all([all_16s, for_pars_16s, 30000], 
                                  [all_chim, for_pars_chim, 25000], 
                                  [all_gen_gnr, for_pars_gen, 5616450, 5000])

'''
3) Create reference db (.csv) containing id,meta,class for sequences choosed to be parsed.
'''
def create_db():
    csv_processing.create_ref_db([info_16s_and_chim, info_gen_gnr], [for_pars_16s, for_pars_chim, for_pars_gen], ref_db)

'''
4) Process parsed files using linux terminal.
   $ echo '\n' >> 16s.csv    (add '\n' to the end of 16s, chimeras and genomic files if none)
   $ cat 16s.csv chimeras.csv genomic_generated.csv > all.csv    (concat all files)
   $ shuf all.csv -o all_shuffled.csv    (shuffle)
   $ echo '\n' >> all_shuffled.csv    (add '\n')
'''

'''
5) Split shuffled file into train:split by split number and add nn1 collumn with values train/test to reference db. 
'''
def for_nn1():
    select_for_nn1.select_train_and_valid(all_shuffled, ref_db, for_nn1_train, for_nn1_valid, 45000)