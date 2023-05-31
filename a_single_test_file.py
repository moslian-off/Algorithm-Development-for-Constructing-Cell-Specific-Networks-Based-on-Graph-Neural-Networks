import os

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import scanpy as sc
import torch
import torch_geometric.utils as utils
import umap
from sklearn.cluster import KMeans
from tqdm import tqdm


def draw_csn(bool_matrix, name):
    global genes_name
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
    labels = {n: genes_name[n] for n in draw_nodes}
    nx.draw_networkx_labels(Gt, pos, labels, font_size=10)
    plt.savefig('G_' + name + '.png')
    plt.close()


def cluster_mean_graph(datas_list, filename, cutoff):
    global genes_name
    os.chdir('Datas_Chutype')
    datas = [torch.load(data_name) for data_name in datas_list]
    os.chdir('..')
    Gs = [utils.to_networkx(data) for data in datas]
    adj_mats = [nx.to_numpy_matrix(G) for G in Gs]
    sum_adj_mat = np.sum(adj_mats, axis=0)
    mean_adj_mat = sum_adj_mat / len(adj_mats)
    mean_adj_mat[mean_adj_mat > cutoff] = 1
    mean_adj_mat[mean_adj_mat < cutoff] = 0
    draw_csn(mean_adj_mat, filename)


def visualize_data(data_name, folder_name):
    global genes_name
    os.chdir('Datas_Chutype')
    data = torch.load(data_name)
    os.chdir('..')
    if os.path.exists(folder_name):  # 如果文件夹已存在
        os.chdir(folder_name)  # 进入文件夹
    else:  # 如果文件夹不存在
        os.mkdir(folder_name)  # 创建文件夹
        os.chdir(folder_name)  # 进入文件夹

    # 创建画布对象
    fig, ax = plt.subplots(figsize=(8, 6))
    plt.rcParams['font.size'] = 12  # 设置字体大小
    plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

    G = nx.Graph()
    for i in range(data.num_nodes):
        G.add_node(i, label=genes_name[i])
    edge_index = data.edge_index.numpy()
    for i in range(edge_index.shape[1]):
        src, dst = edge_index[0][i], edge_index[1][i]
        G.add_edge(src, dst)
    G.remove_nodes_from(list(nx.isolates(G)))

    # 使用 circular_layout 或 spectral_layout 函数生成 layout
    pos = nx.circular_layout(G)
    # pos = nx.spectral_layout(G)

    labels = nx.get_node_attributes(G, 'label')
    nx.draw_networkx(G, pos, labels=labels, with_labels=True, ax=ax)
    plt.savefig(data_name + '.png')
    plt.clf()
    os.chdir('..')
    return G


path = os.getcwd()
filename = 'embedding_by_try_ano_CSN'
X = np.loadtxt(filename + '.txt')
Score_dic = {}
data = sc.read_text('scRNA-data/Chutype/logChumarker.txt')
data = data.transpose()
cell_type = pd.read_csv('scRNA-data/Chutype/chutypectname.txt', sep=' ')  # The observation labeled with cell types.
cell_type.index = data.obs.index
data.obs = cell_type
genes_name = np.array(data.var.index)
# adata = sc.datasets.pbmc68k_reduced()
# adata.obs['cell_type'] = adata.obs['louvain']
# sc.pp.pca(adata, n_comps=100)
# pca_features = adata.obsm['X_pca']
# pca_features = pca_features[:, :100]
# feature_indices = np.arange(100)
# X_new = adata.X[:, feature_indices]
# var_new = adata.var.iloc[feature_indices, :]
# data = anndata.AnnData(X=X_new, obs=adata.obs, var=var_new)
# values = data.obs['cell_type'].value_counts()
new_values = pd.read_csv('new_values.csv')
n_types = new_values.shape[0]
labels = []
for i in range(n_types):
    temp = [i] * new_values['cell_type'][i]
    labels = labels + temp

label_dict = {label: i for i, label in enumerate(np.unique(labels))}
colors = [label_dict[label] for label in labels]

reducer = umap.UMAP(random_state=42, n_components=2)
X_umap = reducer.fit_transform(X)
plt.figure(figsize=(15, 14))
plt.scatter(X_umap[:, 0], X_umap[:, 1], c=colors, cmap='tab10', s=5)
plt.gca().set_aspect('equal', 'datalim')
cbar = plt.colorbar(boundaries=np.arange(n_types + 1) - 0.5)
cbar.set_ticks(np.arange(n_types))
cbar.set_ticklabels(new_values.iloc[:, 0].tolist())
plt.title('UMAP')
plt.savefig(filename + '_umap.png')
plt.close()
cluster_i_indices = []
km_umap = KMeans(n_clusters=9, random_state=42)
umap_labels = km_umap.fit_predict(X_umap)
for i in range(9):
    cluster_i = np.where(umap_labels == i)[0]
    cluster_i_indices.append(cluster_i)

datas = []
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
os.chdir('Datas_Chutype')
filelist = os.listdir(os.getcwd())
for i in range(len(filelist)):
    data = torch.load(filelist[i])
    data.to(device)
    datas.append(data)
os.chdir('..')

datas_by_cluster = []
for i in range(9):
    data_i = [filelist[j] for j in cluster_i_indices[i]]
    datas_by_cluster.append(data_i)

# for k in range(9):
#     for i in range(len(datas_by_cluster[k])):
#         visualize_data(datas_by_cluster[k][i], 'cluster' + str(k))
#         plt.close()

for k in range(9):
    cluster_mean_graph(datas_by_cluster[k], 'cluster' + str(k), 0.3)
    plt.close()
