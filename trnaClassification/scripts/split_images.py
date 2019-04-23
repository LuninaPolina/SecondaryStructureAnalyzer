import pandas as pd
from PIL import Image
import os
import glob

#change here
src = '../images80/'
db_file = '../ref_db.csv'
classes = ['a', 'b', 'f', 'p'] # ['p', 'e']


def make_dirs(dirs, subdirs):
    for d in dirs:
        for sd in subdirs:
            if not os.path.exists(d + sd):
                os.makedirs(d + sd)


make_dirs([src], ['train', 'valid', 'test'])
make_dirs([src + 'train/', src + 'valid/', src + 'test/'], classes)

files = [f for f in glob.glob(src + '/*.bmp')]
db = pd.read_csv(db_file)
for f in files:
    i_d = int(f.split('\\')[-1].split('.')[0])
    nn1 = db.loc[db['id'] == i_d].values[0][3]
    cls = db.loc[db['id'] == i_d].values[0][2]
    img = Image.open(f)
    img.save(src + nn1 + '/' + cls + '/' + str(i_d) + '.bmp')