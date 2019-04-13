import os
import glob
from PIL import Image
import csv

src = '/home/polina/Desktop/origin'    #change paths
resized_dir = '/home/polina/Desktop/resized'
out_dir = '/home/polina/Desktop/splitted'
db_file = '/home/polina/Desktop/ref_db.csv'


def mkdir(name):
    if not os.path.exists(name):
        os.makedirs(name)

def resize(n):
    mkdir(resized_dir)
    files = [f for f in glob.glob(src + '/*.bmp')]
    for f in files:
        img = Image.open(f)
        layer = Image.new('RGB', (n, n), (255,255,255))
        layer.paste(img)
        layer.save(f.replace('/origin', '/resized'))

        
def split():
    mkdir(out_dir + '/train')
    mkdir(out_dir + '/test')
    mkdir(out_dir + '/valid')
    out = open(db_file, mode='w', newline='')
    db = csv.writer(out, delimiter=',')
    db.writerow(['id', 'filename', 'class', 'nn1'])
    files = [f for f in glob.glob(resized_dir + '/*.bmp')]
    cnt = 0
    for f in files:
        img = Image.open(f)
        cls = f.split('_')[2]
        if 'TEST' in f:
            nn1 = 'valid'
        if 'TRAIN' in f:
            nn1 = 'train'
        if 'VALID' in f:
            nn1 = 'test'
        img.save(out_dir + '/' + nn1 + '/' + str(cnt) + '.bmp')
        f = f.split('/')[-1].split('.')[0]
        db.writerow([str(cnt), f, cls, nn1])
        cnt += 1

resize(220)
split()
     
    
    
