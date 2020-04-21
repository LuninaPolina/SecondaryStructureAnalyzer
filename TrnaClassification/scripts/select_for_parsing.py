import pandas as pd
from Bio import SeqIO
import random as rn

#change here
path = '../' 
abfp = [path + 'all/a.fasta', path + 'all/b.fasta', path + 'all/f.fasta', path + 'all/p.fasta']
db_file = path + 'all/ref_db.csv'


def replace_nucls(seq):
    map_dict = {'W': ['A', 'T'], 'S': ['C', 'G'], 'M': ['A', 'C'], 'K': ['G', 'T'],
         'R': ['A', 'G'], 'Y': ['C', 'T'], 'B': ['C', 'G', 'T'], 'D': ['A', 'G', 'T'],
         'H': ['A', 'C', 'T'], 'V': ['A', 'C', 'G'], 'N': ['A', 'C', 'G', 'T'], 'Z': ['']}
    for nucl in seq:
        if nucl in map_dict.keys():
            seq = seq.replace(nucl, rn.choice(map_dict[nucl]))
    return seq

def align(seq, symbol='D', length=220):
    if len(seq) < length:
        return seq + symbol * (length - len(seq))
    else:
        return seq[:length]

def select_for_parsing():
    abfp2 = [path + 'for_parsing/a.fasta', path + 'for_parsing/b.fasta', path + 'for_parsing/f.fasta', path + 'for_parsing/p.fasta']
    train_ids, val_ids, test_ids = [], [], []
    for i in range(4):
        cnt = 0
        with open(abfp2[i], 'w') as out:
            inp = list(SeqIO.parse(open(abfp[i]), 'fasta'))
            rn.shuffle(inp)
            for sample in inp:
                meta, seq = sample.description, str(sample.seq)
                if cnt < 2000:
                    train_ids.append(meta)
                    out.write('>' + meta + '\n' + align(replace_nucls(seq)) + '\n')
                if cnt >= 2000 and cnt < 2250:
                    val_ids.append(meta)
                    out.write('>' + meta + '\n' + align(replace_nucls(seq)) + '\n')
                if cnt >= 2250 and cnt < 3000:
                    test_ids.append(meta)
                    out.write('>' + meta + '\n' + align(replace_nucls(seq)) + '\n')
                cnt += 1
    db = pd.read_csv(db_file)
    flags = ['valid' if str(row['id']) in val_ids else 'train' if str(row['id']) in train_ids else 'test' if str(row['id']) in test_ids else 'none' for index, row in db.iterrows()]
    db['nn1'] = flags
    db.to_csv(db_file, index=False)

select_for_parsing()               