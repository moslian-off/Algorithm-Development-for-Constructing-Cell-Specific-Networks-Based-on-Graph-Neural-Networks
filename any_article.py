import os

import locCSN
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import scanpy as sc
from scipy.stats import norm


def draw_csn(bool_matrix, name):
    global data
    Gt = nx.Graph()
    g = bool_matrix.shape[0]
    for i in range(g):
        for j in range(i + 1, g):
            if bool_matrix[i][j] == 1:
                Gt.add_edge(i, j)
    draw_nodes = [n for n in Gt.nodes()]
    pos = nx.spring_layout(Gt)
    nx.draw_networkx_edges(Gt, pos)
    nx.draw_networkx_nodes(Gt, pos, nodelist=draw_nodes, node_size=10)
    labels = {n: data.var.index[n] for n in draw_nodes}
    nx.draw_networkx_labels(Gt, pos, labels, font_size=10)
    plt.savefig('G_' + name + '.png')
    plt.close()


# Set path to data
os.chdir('scRNA-data/Chutype/')

# read in Chutype dataset
data = sc.read_text('logChumarker.txt')
data = data.transpose()  # 1018 cells * 51 genes

cell_type = pd.read_csv('chutypectname.txt', sep=' ')
data.obs = cell_type  # The observation are labeled with cell types.

os.chdir('../..')

data_dec = data[data.obs.cell_type == "DEC",]
X_dec = data_dec.X.transpose()
data_npc = data[data.obs.cell_type == 'NPC',]
X_npc = data_npc.X.transpose()
data_ec = data[data.obs.cell_type == 'EC',]
X_ec = data_ec.X.transpose()
data_h1 = data[data.obs.cell_type == 'H1',]
X_h1 = data_h1.X.transpose()
data_h9 = data[data.obs.cell_type == 'H9',]
X_h9 = data_h9.X.transpose()
data_hff = data[data.obs.cell_type == 'HFF',]
X_hff = data_hff.X.transpose()
data_tb = data[data.obs.cell_type == 'TB',]
X_tb = data_tb.X.transpose()

corr_dec = np.corrcoef(X_dec)
corr_npc = np.corrcoef(X_npc)
corr_ec = np.corrcoef(X_ec)
corr_h1 = np.corrcoef(X_h1)
corr_h9 = np.corrcoef(X_h9)
corr_hff = np.corrcoef(X_hff)
corr_tb = np.corrcoef(X_tb)

np.fill_diagonal(corr_dec, 0)
np.fill_diagonal(corr_npc, 0)
np.fill_diagonal(corr_ec, 0)
np.fill_diagonal(corr_h1, 0)
np.fill_diagonal(corr_h9, 0)
np.fill_diagonal(corr_hff, 0)
np.fill_diagonal(corr_tb, 0)

csn_dec = locCSN.csn(X_dec, dev=True)
csn_npc = locCSN.csn(X_npc, dev=True)
csn_ec = locCSN.csn(X_ec, dev=True)
csn_h1 = locCSN.csn(X_h1, dev=True)
csn_h9 = locCSN.csn(X_h9, dev=True)
csn_hff = locCSN.csn(X_hff, dev=True)
csn_tb = locCSN.csn(X_tb, dev=True)

# plt.imshow(csn_dec[0].toarray(), vmin=-6, vmax=6, cmap='coolwarm')
# plt.title('DEC one cell', fontweight="bold")
# plt.colorbar()
# plt.savefig('dec_one_cell.png')
#
# plt.imshow(csn_npc[0].toarray(), vmin=-6, vmax=6, cmap='coolwarm')
# plt.title('NPC one cell', fontweight="bold")
# plt.savefig('npc_one_cell.png')
#
# plt.imshow(csn_ec[0].toarray(), vmin=-6, vmax=6, cmap='coolwarm')
# plt.title('EC one cell', fontweight="bold")
# plt.savefig('ec_one_cell.png')
#
# plt.imshow(csn_h1[0].toarray(), vmin=-6, vmax=6, cmap='coolwarm')
# plt.title('H1 one cell', fontweight="bold")
# plt.savefig('h1_one_cell.png')
#
# plt.imshow(csn_h9[0].toarray(), vmin=-6, vmax=6, cmap='coolwarm')
# plt.title('H9 one cell', fontweight="bold")
# plt.savefig('h9_one_cell.png')
#
# plt.imshow(csn_hff[0].toarray(), vmin=-6, vmax=6, cmap='coolwarm')
# plt.title('HFF one cell', fontweight="bold")
# plt.savefig('hff_one_cell.png')
#
# plt.imshow(csn_tb[0].toarray(), vmin=-6, vmax=6, cmap='coolwarm')
# plt.title('TB one cell', fontweight="bold")
# plt.savefig('tb_one_cell.png')

csn_mean_dec = (np.mean(np.vstack(csn_dec), axis=0))[0]
csn_mean_npc = (np.mean(np.vstack(csn_npc), axis=0))[0]
csn_mean_ec = (np.mean(np.vstack(csn_ec), axis=0))[0]
csn_mean_h1 = (np.mean(np.vstack(csn_h1), axis=0))[0]
csn_mean_h9 = (np.mean(np.vstack(csn_h9), axis=0))[0]
csn_mean_hff = (np.mean(np.vstack(csn_hff), axis=0))[0]
csn_mean_tb = (np.mean(np.vstack(csn_tb), axis=0))[0]

csn_bool_dec = np.where(csn_mean_dec.toarray() > norm.ppf(0.9), 1, 0)
csn_bool_npc = np.where(csn_mean_npc.toarray() > norm.ppf(0.9), 1, 0)
csn_bool_ec = np.where(csn_mean_ec.toarray() > norm.ppf(0.85), 1, 0)
csn_bool_h1 = np.where(csn_mean_h1.toarray() > norm.ppf(0.85), 1, 0)
csn_bool_h9 = np.where(csn_mean_h9.toarray() > norm.ppf(0.75), 1, 0)
csn_bool_hff = np.where(csn_mean_hff.toarray() > norm.ppf(0.85), 1, 0)
csn_bool_tb = np.where(csn_mean_tb.toarray() > norm.ppf(0.85), 1, 0)

draw_csn(csn_bool_dec, 'dec')
draw_csn(csn_bool_npc, 'npc')
draw_csn(csn_bool_ec, 'ec')
draw_csn(csn_bool_h1, 'h1')
draw_csn(csn_bool_h9, 'h9')
draw_csn(csn_bool_hff, 'hff')
draw_csn(csn_bool_tb, 'tb')
