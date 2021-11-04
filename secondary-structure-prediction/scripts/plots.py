'''Functions for drawing different plots used in our papers'''

import matplotlib.pyplot as plt


m = 'o'
mw = 1.5


def plot_f1(out_file):
    hot, spot, rg, struct, ipknot, knotty  = 66.3, 72, 66.4, 65.9, 66.0, 67.2
    model_x = [10, 20, 30, 40, 50, 60, 70, 80, 90] 
    model_y_new = [58.5, 62.0, 63.1, 66.1, 66.7, 69.3, 71.9, 70.8, 72.4]
    fig, ax = plt.subplots()
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.plot(33, struct,  marker=m, markeredgewidth=mw, linestyle='None', color='red')
    plt.plot(37, ipknot,  marker=m, markeredgewidth=mw, linestyle='None', color='orange')
    plt.plot(43, hot,  marker=m, markeredgewidth=mw,linestyle='None', color='gold')
    plt.plot(47, rg,  marker=m, markeredgewidth=mw, linestyle='None', color='green')
    plt.plot(55, knotty,  marker=m, markeredgewidth=mw, linestyle='None', color='skyblue')
    plt.plot(85, spot,  marker=m, markeredgewidth=mw, linestyle='None', color='blue')
    plt.plot(model_x, model_y_new,marker=m, markeredgewidth=mw, linestyle='dashed', color='indigo', linewidth=0.5)
    plt.legend(['RNAstructure', 'Ipknot',  'HotKnots', 'pknotsRG', 'Knotty', 'SPOT-RNA', 'New-model'], loc='lower right', handlelength=0, borderpad=0.7)
    plt.axis((5, 95, 58, 73))
    plt.xlabel('Data percent for new model training')
    plt.ylabel('F1 score')
    plt.savefig(out_file, dpi=700)
    

def plot_p_r(out_file):
    hot_p, spot_p, rg_p, struct_p, ipknot_p, knotty_p  = 69.3, 73.1, 69.0, 69.5, 71.8, 69.9
    hot_r, spot_r, rg_r, struct_r, ipknot_r, knotty_r  = 64.9, 72.8, 65.3, 64.6, 62.8, 66.1
    model_x = [10, 20, 30, 40, 50, 60, 70, 80, 90]
    model_y_p = [57.7, 61.3, 63.5, 64.9, 64.9, 67.9, 71.2, 68.8, 71.0] 
    model_y_r = [62.1, 65.0, 65.2, 70.1, 70.8, 72.5, 74.6, 75.0, 75.6] 
    fig, ax = plt.subplots()
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False) 
    plt.plot(struct_p, struct_r, marker=m, markeredgewidth=mw, linestyle='None', color='red')
    plt.plot(ipknot_p, ipknot_r, marker=m,markeredgewidth=mw, linestyle='None', color='orange')
    plt.plot(hot_p, hot_r, marker=m, markeredgewidth=mw, linestyle='None', color='gold')
    plt.plot(rg_p, rg_r, marker=m, markeredgewidth=mw, linestyle='None', color='green')
    plt.plot(knotty_p, knotty_r, marker=m, markeredgewidth=mw, linestyle='None', color='skyblue')
    plt.plot(spot_p, spot_r, marker=m, markeredgewidth=mw, linestyle='None', color='blue')
    plt.plot(model_y_p, model_y_r, marker=m, markeredgewidth=mw, linestyle='dashed', color='indigo', linewidth=0.5) 
    plt.text(model_y_p[0] - 0.5, model_y_r[0] - 1, '10% data', color='black', fontsize=7)
    plt.text(model_y_p[8] + 0.5, model_y_r[8] - 0.25, '90% data', color='black', fontsize=7)
    plt.plot([57, 80], [57, 80], color='black', linewidth=0.5)
    plt.legend(['RNAstructure', 'Ipknot',  'HotKnots', 'pknotsRG', 'Knotty', 'SPOT-RNA', 'New-model'], loc='upper left', handlelength=0, borderpad=0.7)
    plt.axis((57, 76, 57, 76))
    plt.xlabel('Precision')
    plt.ylabel('Recall')
    plt.savefig(out_file, dpi=700)


def print_latex_table(seq, img_file):
	out = '\\begin{table}[]\n\\begin{tabular}{' + 'l' * len(seq) + '}\n'
	img = np.array(Image.open(img_file))
	for i in range(len(img)):
	    for j in range(len(img)):
	        if i == j:
	            out += seq[i] + ' '
	        elif j < i:
	             out += '&   '
	        else:
	            if img[i][j] == 0:
	                out += '& ' + str(int(img[i][j] / 255)) + ' '
	            else:
	                out += '& \\textbf{' + str(int(img[i][j] / 255)) + '} '
	    out += ' \\\\[2pt]\n'
	out += '\end{tabular}\n\end{table}'
	print(out)


def plot_distr(in_fasta, out_file):
    distr = {}
    lens = []
    data = open(in_fasta).read().split('\n')
    for i in range(0, len(data) - 1, 2):
        meta, seq = data[i], data[i + 1]
        lens.append(len(seq))
        if len(seq) in distr.keys():
            distr[len(seq)] += 1
        else:
            distr[len(seq)] = 1
    med = statistics.median(lens)
    mean = round(statistics.mean(lens))
    print(med, mean)
    colors = []
    for k in distr.keys():
        if k == med:
            colors.append('red')
        elif k == mean:
            colors.append('blue')
        else:
            colors.append('skyblue')
    plt.figure(figsize=(24,5), dpi=200)
    plt.bar(distr.keys(), distr.values(), color=colors)
    colors = {'Mean':'blue', 'Median':'red'}         
    labels = list(colors.keys())
    handles = [plt.Rectangle((0,0),1,1, color=colors[label]) for label in labels]
    plt.legend(handles, labels, borderpad=0.2, prop={'size': 15})
    plt.xlabel('Sequence length')
    plt.ylabel('Number of samples') 
    plt.xticks([i * 10 for i in range(21)])
    plt.yticks([i * 10 for i in range(9)])
    plt.axis((7, 201, 0, 81))
    plt.savefig(out_file, dpi=200, bbox_inches='tight')


def plot_curves(log_files, out_dir):
    for f in log_files:
        name = f.split('/')[-1].split('_')[1] + '% train data'
        train, test = [], []
        log = open(f).read().split('\n')[1:-1]
        for line in log:
            line = line.split(',')
            train.append(round(float(line[1]) * 100))
            test.append(round(float(line[5]) * 100))
        x = [i for i in range(1, len(train) + 1)]
        plt.plot(x, train)
        plt.plot(x, test)
        plt.legend(['F1-train', 'F1-test'], title = name, loc='lower right')
        plt.axis((0, 200, 40, 85))
        plt.savefig(out_dir + name + '.png', dpi=300)
        plt.show()


def estimate(img_true, img_pred):                       
    size = len(img_true)
    tw, fw, fb = 0, 0, 0 
    for i in range(size):
        for j in range(i + 1, size):  
            if img_true[i][j] != img_pred[i][j] and i != j:
                if int(img_pred[i][j]) == 0:
                    fb += 1
                else:
                    fw += 1
            elif img_true[i][j] == img_pred[i][j] and i != j:
                if int(img_true[i][j]) == 255:
                    tw += 1
    prec = tw / (tw + fw + 0.00001)
    rec = tw / (tw + fb + 0.00001)
    f1 = 2 * (prec * rec) / (prec + rec + 0.00001)  
    return prec, rec, f1


def plot_f1_violins(true_dir, pred_dirs, val_ids, out_file):
    files_true = glob.glob(true_dir + '*.png')
    data = [[] for pd in pred_dirs]
    for ft in files_true:
        if ft.split('/')[-1].split('.')[0] in val_ids.split(' '):
            img_t = np.array(Image.open(ft))
            for i in range(len(pred_dirs)):
                fp = ft.replace(true_dir, pred_dirs[i])
                try:
                    img_p = np.array(Image.open(fp))                 
                    f1 = estimate(img_t, img_p)[2]
                    data[i].append(f1 * 100)
                except: ()
    data = list(map(lambda x: sorted(x), data))  
    for i in range(len(data)):
        print(pred_dirs[i].split('/')[-1], statistics.mean(data[i]), statistics.median(data[i]))
    colors = ['slateblue', 'dodgerblue', 'skyblue', 'springgreen', 'gold', 'darkorange', 'tomato']
    plt.figure(figsize=(12,5), dpi=200)
    vp = plt.violinplot(data, showmeans=True, showmedians=True, showextrema=False)
    vp['cmeans'].set_edgecolor('blue')
    vp['cmedians'].set_edgecolor('red')
    for i in range(7):
        vp['bodies'][i].set_facecolor(colors[i])
        vp['bodies'][i].set_alpha(0.3)
    plt.xticks(range(8), ['', 'Genegram-pks', 'SPOT-RNA', 'Ipknot', 'Knotty', 'RNAstructure', 'PknotsRG', 'HotKnots'])
    plt.yticks([i * 10 for i in range(11)])
    plt.axis((0, 8, 0, 105)) 
    plt.ylabel('F1-score')
    plt.xlabel(' ')
    colors = {'Mean':'blue', 'Median':'red'}         
    labels = list(colors.keys())
    handles = [plt.Rectangle((0,0),1,1, color=colors[label]) for label in labels]
    plt.legend(handles, labels, loc='lower left', borderpad=0.2)
    plt.savefig(out_file, dpi=200, bbox_inches='tight')
    plt.show()


def plot_prec_rec(true_dir, pred_dirs, val_ids, out_file):
    files_true = glob.glob(true_dir + '*.png')
    data_p = [[] for pd in pred_dirs]
    data_r = [[] for pd in pred_dirs]
    for ft in files_true:
        if ft.split('/')[-1].split('.')[0] in val_ids.split(' '):
            img_t = np.array(Image.open(ft))
            for i in range(len(pred_dirs)):
                fp = ft.replace(true_dir, pred_dirs[i])
                try:
                    img_p = np.array(Image.open(fp))                  
                    prec, rec, f1 = estimate(img_t, img_p)
                    data_p[i].append(prec * 100)
                    data_r[i].append(rec * 100)
                except: () 
    colors = ['slateblue', 'dodgerblue', 'skyblue', 'springgreen', 'gold', 'darkorange', 'tomato']   
    plt.figure(figsize=(5,5), dpi=200)
    for i in range(7):
        plt.plot(statistics.mean(data_p[i]), statistics.mean(data_r[i]), marker='D', markeredgewidth=3, linestyle='None', color=colors[i])
    plt.xticks([i * 5 for i in range(21)])
    plt.yticks([i * 5 for i in range(21)])
    plt.plot([55, 80], [55, 80], color='black', linewidth=0.5)
    plt.legend(['Genegram-pks', 'SPOT-RNA', 'Ipknot', 'Knotty', 'RNAstructure', 'PknotsRG', 'HotKnots'], loc='upper left', borderpad=0.2)
    plt.axis((55, 80, 55, 80))
    plt.gca().set_aspect('equal')
    plt.xlabel('Precision')
    plt.ylabel('Recall')

    plt.savefig(out_file, dpi=200, bbox_inches='tight')