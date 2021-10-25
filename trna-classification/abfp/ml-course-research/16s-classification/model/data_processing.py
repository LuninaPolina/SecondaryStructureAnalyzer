from Bio import SeqIO
import pandas as pd
import numpy as np

# Extract species from taxon
def get_species_by_description(desc):
    tax = desc.split(" ")[1].split(";")
    if tax[0] == "Bacteria" or tax[0] == "Archaea":
        return tax[0]
    elif "Fungi" in tax:
        return "Fungi"
    elif tax[1] == "Archaeplastida":
        return "Plantae"
    else:
        return "other"
    

def slice_seq(seq, step = 160):
    i = 0
    res = []
    while i < len(seq):
        s = seq[i:i+220]
        if len(s) < 220: # Drop last short sequence
            break
        r = []
        # Convert string to tensor
        for l in s:
            if l == "A": r.append("2") 
            elif l == "C": r.append("3")
            elif l == "G": r.append("5")
            elif l == "U": r.append("7")
            else: r.append("0")
        res.append(np.asarray(r).astype(np.float32))
        i += step
    return np.stack( res, axis=0 )


def get_data(path, seqs_num = 8):
    sample_limit = 9000
    found_samples = {"Bacteria" : 0, "Archaea" : 0, "Fungi" : 0, "Plantae" : 0}

    df = pd.DataFrame(columns = ["Id", "Seqs", "Label"])
    row_list = []
    df_ref = pd.DataFrame(columns = ["meta", "class", "nn1"])
    for i, record in enumerate(SeqIO.parse(path, "fasta")): 
        #if i % 1000 == 0: # Progress
            #print(i)
        species = get_species_by_description(record.description)
        id = record.id   
        row_list.append({"ref_id" : id, "meta" : record.description, "class" : species, "nn1" : "none"})
        if species == "other": # Drop redundant samples
            continue
        
        if found_samples[species] >= sample_limit:
            if len(df) >= sample_limit * 4:
                break
            else:   
                continue
        seqs = slice_seq(record.seq)
        if seqs.shape[0] >= seqs_num: # Drop excessive sequences
            seqs = seqs[:seqs_num]
            df = df.append({"Id" : id, "Seqs" : seqs, "Label" : species}, ignore_index = True)
            found_samples[species] += 1 
    df_ref = pd.DataFrame(row_list)
    df_ref.index = df_ref.index.rename('id')
    return df, df_ref

def process_file(path):
    df = pd.read_pickle(path)
    # Convert labels to vectors
    x = df.Seqs.values
    y_t = df.Label.values
    y = []

    for label in y_t:
        t = None
        if label == "Archaea":
            t = [1, 0, 0, 0]
        elif label == "Plantae":
            t = [0, 1, 0, 0]
        elif label == "Fungi":
            t = [0, 0, 1, 0]
        elif label == "Bacteria":
            t = [0, 0, 0, 1]
        y.append(np.asarray(t).astype(np.float32))

    y=np.asarray(y)

    x = np.stack( x, axis=0 ) # Data transformation

    return x,y

if __name__ == "__main__":
    df, df_ref = get_data("../../../../../data/SILVA_138.1_SSURef_NR99_tax_silva.fasta")
    x = df.iloc[:, :-1]
    y = df.Label.values
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(x,y)
    df_ref.loc[df_ref.ref_id.isin(X_train.Id), "nn1"] = "train"
    df_ref.loc[df_ref.ref_id.isin(X_test.Id), "nn1"] = "test"
    df_ref = df_ref.drop("ref_id", axis = 1)
    df_train = X_train.Seqs
    df_train["Label"] = y_train
    df.to_pickle("./data/train.pkl")
    df_test = X_test.Seqs
    df_test["Label"] = y_test
    df.to_pickle("./data/test.pkl")
    df_ref.to_csv("./data/ref_db.csv", sep = "\t")
