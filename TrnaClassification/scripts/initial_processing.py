import csv
from Bio import SeqIO

#change here
path = '../'
db1 = [path + 'in/a.fasta', path + 'in/b.fasta', path + 'in/f.fasta', path + 'in/p.fasta']
db2 = [path + 'in/a2.fasta', path + 'in/b2.fasta', path + 'in/f2.fasta', path + 'in/p2.fasta']
db_file = path + '../ref_db.csv'


p_names = ['Brachypodium_distachyon', 'Physcomitrella_patens', 'Sorghum_bicolor', 'Vitis_vinifera', 'Zea_mays',
           'Arabidopsis_thaliana', 'Glycine_max', 'Medicago_truncatula', 'Oryza_sativa', 'Populus_trichocarpa']
f_names = ['Aspergillus_fumigatus', 'Candida_glabrata', 'Cryptococcus_neoformans', 'Debaryomyces_hansenii',
           'Encephalitozoon_cuniculi', 'Eremothecium_gossypii', 'Kluyveromyces_lactis', 'Magnaporthe_oryzae',
           'Saccharomyces_cerevisiae', 'Saccharomyces_cerevisiae', 'Schizosaccharomyces_pombe', 'Yarrowia_lipolytica']

def split_euk():
    inp = list(SeqIO.parse(open(path + 'in/euk.fasta'), 'fasta'))
    fungi_file = db2[2]
    plant_file = db2[3]
    with open(fungi_file, 'w') as out_f, open(plant_file, 'w') as out_p:
        for sample in inp:
             meta, seq = sample.description, str(sample.seq)
             if any(x in meta for x in p_names):
                 out_p.write('>' + meta + '\n' + seq + '\n')
             if any(x in meta for x in f_names):
                 out_f.write('>' + meta + '\n' + seq + '\n')
             
def get_all():
    abfp = [path + 'all/a.fasta', path + 'all/b.fasta', path + 'all/f.fasta', path + 'all/p.fasta']
    cnt = 0
    with open(db_file, 'w') as db_out:
        db = csv.writer(db_out, delimiter=',')
        db.writerow(['id', 'meta', 'class'])
        for i in range(4):
            with open(abfp[i], 'w') as out:
                inp1 = list(SeqIO.parse(open(db1[i]), 'fasta'))
                inp2 = list(SeqIO.parse(open(db2[i]), 'fasta'))
                for sample in inp1:
                    meta, seq = sample.description, str(sample.seq)
                    if not 'andidatus' in meta and not 'nclassified' in meta:
                        cls = abfp[i].split('/')[-1].split('.')[0]
                        db.writerow([str(cnt), meta, cls])
                        seq = ''.join(list(filter(lambda x: x == x.upper(), seq)))
                        out.write('\n>' + str(cnt) + '\n' + seq)
                        cnt += 1
                for sample in inp2:
                    meta, seq = sample.description, str(sample.seq)
                    if not 'andidatus' in meta and not 'nclassified' in meta:
                        cls = abfp[i].split('/')[-1].split('.')[0]
                        db.writerow([str(cnt), meta, cls])
                        out.write('\n>' + str(cnt) + '\n' + seq)
                        cnt += 1

split()
get_all()