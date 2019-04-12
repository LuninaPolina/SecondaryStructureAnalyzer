import pandas as pd
from Bio import SeqIO

def create_ref_db(csv_inp_files, fasta_inp_files, db_file):
    ids = []
    for file in fasta_inp_files:
        inp = SeqIO.parse(open(file), 'fasta')
        for sample in inp:
            ids.append(sample.description)
    data1 = pd.read_csv(csv_inp_files[0], header=None)
    data1 = data1[data1[0].isin(ids)]
    data2 = pd.read_csv(csv_inp_files[1], header=None)
    data2 = data2[data2[0].isin(ids)]
    db = pd.concat([data1, data2])
    db.columns = ['id', 'meta', 'class']
    db.to_csv(db_file, index=False)