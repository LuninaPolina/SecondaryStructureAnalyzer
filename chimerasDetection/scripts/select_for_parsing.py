from Bio import SeqIO
from random import shuffle
from Bio.SeqIO.FastaIO import FastaWriter

def select_from_small_file(args):
    inp_file, db_inp_file, db_out_file, out_file, num = args
    inp = list(SeqIO.parse(open(inp_file), 'fasta'))
    shuffle(inp)
    writer = FastaWriter(open(out_file, 'w'), wrap=0)
    writer.write_file(inp[:num])

def select_from_big_file(args):
    inp_file, out_file, total_num, num = args
    selection = []
    with open (out_file, 'w') as out:
        inp = SeqIO.parse(open(inp_file), 'fasta')
        fst_line = True
        cnt, cnt_out = 0, 0
        for sample in inp:
            if cnt % int(total_num / num) == 0 and cnt_out < num:
                selection.append(sample)
                cnt_out += 1
            cnt += 1
        shuffle(selection)
        for sample in selection:
            meta, seq = sample.description, str(sample.seq)
            if fst_line:
                out.write('>' + meta + '\n' + seq)
                fst_line = False
            else:
                out.write('\n>' + meta + '\n' + seq)      

def select_all(args_16s, args_chim, args_genomic):
    select_from_small_file(args_16s)
    select_from_small_file(args_chim)
    select_from_big_file(args_genomic)