import csv
import pandas as pd

def select_train_and_valid(inp_file, db_file, train_file, valid_file, split_pos):
    with open(inp_file, 'r') as inp, open(train_file, 'w') as out_train, open(valid_file, 'w') as out_valid:
        cnt = 1
        val_ids = []
        data = csv.reader(open(inp_file, 'r'))
        train = csv.writer(out_train, delimiter=',')
        valid = csv.writer(out_valid, delimiter=',')
        for line in data:
            if line != []:
                if cnt < split_pos:
                    train.writerow(line)
                else:
                    valid.writerow(line)
                    val_ids.append(line[0][1:])
                cnt += 1
    db = pd.read_csv(db_file)
    flags = ['valid' if str(row['id']) in val_ids else 'train' for index, row in db.iterrows()]
    db['nn1'] = flags
    db.to_csv(db_file, index=False)