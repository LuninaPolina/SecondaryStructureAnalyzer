import pandas as pd
from Bio import SeqIO
import random as rn

#change here
seq_file = '/home/polina/Desktop/all_seq.fasta'
db_file = '/home/polina/Desktop/ref_db.csv'
train_file = '../train.csv'
valid_file = '../valid.csv'
test_file = '/home/polina/Desktop/test.csv'

train, valid, test = [], [], []

db = pd.read_csv(db_file)

def align(seq, symbol='D', length=220):
    if len(seq) < length:
        return seq + symbol * (length - len(seq))
    else:
        return seq[:length]

with open(seq_file, 'r') as f:
    inp = list(SeqIO.parse(f, 'fasta'))
    for sample in inp:
        meta, seq = sample.description, str(sample.seq)
        nn1 = db.loc[db['id'] == int(meta)].values[0][3]
        if nn1 == 'train':
            train.append(sample)
        elif nn1 == 'valid':
            valid.append(sample)
        else:
            test.append(sample)

rn.shuffle(train)
rn.shuffle(valid)
rn.shuffle(test)

def write_selection(file, data):
    with open(file, 'w') as out:
        for sample in data:
            meta, seq = sample.description, align(str(sample.seq))
            out.write('>' + meta + ',' + ','.join(seq) + '\n')

#write_selection(train_file, train)
#write_selection(valid_file, valid)
write_selection(test_file, test)
