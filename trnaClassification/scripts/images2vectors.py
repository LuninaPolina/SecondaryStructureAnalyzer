import glob
from PIL import Image
import numpy
from bitstring import BitArray
import csv
import random

#change here
train_dir = '../train' 
valid_dir = '../valid'
test_dir = '../test'

train_out = '../train.csv'
valid_out = '../valid.csv'
test_out = '../test.csv'


def process_dir(src, out_file):
    with open(out_file, mode='w', newline='') as out:
        writer = csv.writer(out, delimiter=',')
        cnt = 0
        files = glob.glob(src + '/*.bmp')
        for f in files:
            cnt += 1
            im = Image.open(f)
            mtx = numpy.array(im)
            bits = []
            ints = []
            for i in range(len(mtx)):
                for j in range(i + 1):
                    if mtx[i, j][0] == 255:
                        bits.append(0)
                    else:
                        bits.append(1)   
            im.close()
            for i in range(int(len(bits)/32)):      
                b = BitArray(bits[32*i:32*(i+1)])
                ints.append(b.uint)
            name = '>' + f.split('/')[-1].split('.')[0]
            line = [name] + ints       
            writer.writerow(line)
            if cnt % 100 == 0:
                print(str(cnt) + ' done')
        
process_dir(train_dir, train_out)
process_dir(valid_dir, valid_out)
process_dir(test_dir, test_out)