import os

import locCSN
import numpy as np
import pandas as pd
import scanpy as sc
import torch
from scipy.stats import norm
from torch_geometric.data import Data

path = os.getcwd()
ppf = 0.95


def get_num(values, i, j):
    r = 0
    for k in range(i):
        r = r + values[i]
    return r + j


# Set path to data

data = sc.read_text('scRNA-data/Chutype/logChumarker.txt')
data = data.transpose()
cell_type = pd.read_csv('scRNA-data/Chutype/chutypectname.txt', sep=' ')  # The observation labeled with cell types.
cell_type.index = data.obs.index
data.obs = cell_type
# adata = sc.datasets.pbmc68k_reduced()
# adata.obs['cell_type'] = adata.obs['louvain']
# sc.pp.pca(adata, n_comps=100)
# pca_features = adata.obsm['X_pca']
# pca_features = pca_features[:, :100]
# feature_indices = np.arange(100)
# X_new = adata.X[:, feature_indices]
# var_new = adata.var.iloc[feature_indices, :]
# data = anndata.AnnData(X=X_new, obs=adata.obs, var=var_new)
values = data.obs['cell_type'].value_counts()
n_types = values.shape[0]
values.to_csv('values.csv', encoding='gbk')
print(f"{n_types} types in total")

# labels = []
# for i in range(n_types):
#     temp = [i] * values[i]
#     labels = labels + temp
#
# label_dict = {label: i for i, label in enumerate(np.unique(labels))}
# colors = [label_dict[label] for label in labels]
#
# reducer = umap.UMAP(random_state=42, n_components=2)
# X_umap = reducer.fit_transform(data.X)
# plt.figure(figsize=(15, 14))
# plt.scatter(X_umap[:, 0], X_umap[:, 1], c=colors, cmap='tab10', s=5)
# plt.gca().set_aspect('equal', 'datalim')
# plt.colorbar(boundaries=np.arange(11) - 0.5).set_ticks(np.arange(10))
# plt.title('UMAP')
# plt.savefig('umap_from_scRNA_matrix.png')
# plt.close()
#
#
# pca = PCA(n_components=2)
# plt.figure(figsize=(15, 14))
# embed_2d = pca.fit_transform(data.X)
# plt.scatter(embed_2d[:, 0], embed_2d[:, 1])
# plt.savefig('pca_from_scRNA_matrix.png')
# plt.close()

CSNs = {}
data.obs['CSN'] = None
temp = 0
for i in range(n_types):
    type_name = values.index[i]
    print(f"generating CSNs in type {type_name}, {values[type_name]} cells in this type")
    condition = data.obs.cell_type == type_name
    data_loc = data[condition]
    CSNs_one_type = locCSN.csn(data_loc.X.transpose(), dev=True)
    CSNs[type_name] = CSNs_one_type
    data.obs['CSN'][condition] = [temp + k for k in range(values[type_name])]
    temp = temp + values[type_name]
    print(f"Successfully generating CSNs in type {type_name}")

datas = []

if not os.path.isdir(os.path.join(path, 'Datas_Chutype')):
    os.mkdir('Datas_Chutype')
os.chdir('Datas_Chutype')
sc.pp.normalize_total(data, target_sum=100)
temp = 0
new_values = values.copy()
for i in range(n_types):
    type_name = values.index[i]
    CSNs_one_type = CSNs[type_name]
    csn_mat = [(item > norm.ppf(ppf)).astype(int) for item in CSNs_one_type]
    for j in range(len(csn_mat)):
        CSN = csn_mat[j]
        edge_index = torch.tensor(np.array(CSN.nonzero()), dtype=torch.long)
        y = i
        y = torch.tensor(y)
        idx = get_num(values, i, j)
        x = data.X[data.obs['CSN'] == idx]
        arr = np.arange(CSN.shape[0])
        x_new = torch.tensor(np.concatenate((x.reshape(-1, 1), arr.reshape(-1, 1)), axis=1))
        x_new = x_new.float()
        data_one_graph = Data(x=x_new, edge_index=edge_index, y=y)
        if not data_one_graph.edge_index.shape[1]:
            new_values[i] -= 1
            continue
        datas.append(data_one_graph)
        torch.save(data_one_graph, 'data' + str(temp + j) + '.pt')
    temp = temp + len(CSNs_one_type)

os.chdir('..')
new_values.to_csv('new_values.csv', encoding='gbk')
# model = WeightedGCN(2, 16, n_types)
# optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
#
# for epoch in range(10):
#     for i in tqdm(range(len(datas))):
#         optimizer.zero_grad()
#         output = model(datas[i].x, datas[i].edge_index, datas[i].edge_attr)
#         loss = torch.nn.BCEWithLogitsLoss()(output, datas[i].y.reshape(1, -1))
#         loss.backward()
#         optimizer.step()
