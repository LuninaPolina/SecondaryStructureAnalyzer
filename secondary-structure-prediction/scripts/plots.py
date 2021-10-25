import matplotlib.pyplot as plt

m = 'o'
mw = 1.5
def plot_f1():
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

    plt.savefig('/home/polina/Desktop/plot_f1.png', dpi=700)
    

def plot_p_r():
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

    plt.savefig('/home/polina/Desktop/plot_pr.png', dpi=700)


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