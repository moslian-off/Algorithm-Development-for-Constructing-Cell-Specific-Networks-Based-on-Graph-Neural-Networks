import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import umap
from matplotlib.colors import ListedColormap
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA


def cluster_label_change(umap_label):
    replace_dict = {0: 0, 1: 5, 2: 1, 3: 3, 4: 4, 5: 6, 6: 2}
    new_umap_label = np.vectorize(replace_dict.get)(umap_label)
    return new_umap_label


path = os.getcwd()
filename = 'embedding_by_try_ano_CSN'
X = np.loadtxt(filename + '.txt')
Score_dic = {}
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

km_umap = KMeans(n_clusters=n_types, random_state=42)
umap_labels = km_umap.fit_predict(X_umap)
umap_labels = cluster_label_change(umap_labels)
Silhouette_S = metrics.silhouette_score(X_umap, umap_labels)
dbi = metrics.davies_bouldin_score(X_umap, umap_labels)
ari = metrics.adjusted_rand_score(np.array(colors), umap_labels)
Score_dic['Silhouette_Score'] = [Silhouette_S]
Score_dic['dbi'] = [dbi]
Score_dic['ari'] = [ari]
plt.figure(figsize=(15, 14))
plt.scatter(X_umap[:, 0], X_umap[:, 1], c=umap_labels, cmap='tab10', s=5)
plt.gca().set_aspect('equal', 'datalim')
plt.colorbar(boundaries=np.arange(n_types + 1) - 0.5).set_ticks(np.arange(n_types))
plt.title('UMAP')
plt.savefig(filename + '_umap_kmeans.png')
plt.close()

Score = pd.DataFrame(Score_dic)
Score.to_csv('Score.csv', index=True)

if not os.path.isdir(os.path.join(path, filename + '_umap_by_label')):
    os.mkdir(filename + '_umap_by_label')
os.chdir(filename + '_umap_by_label')
for i in range(len(label_dict)):
    new_color = [0 if j != label_dict[i] else 1 for j in colors]
    reducer = umap.UMAP(random_state=42, n_components=2)
    X_umap = reducer.fit_transform(X)
    plt.figure(figsize=(15, 14))
    cmap = ListedColormap(['gray', 'blue'])
    alpha = [1 if c == 1 else 0.2 for c in new_color]
    plt.scatter(X_umap[:, 0], X_umap[:, 1], c=new_color, cmap=cmap, alpha=alpha)
    plt.gca().set_aspect('equal', 'datalim')
    plt.title('UMAP')
    plt.savefig(filename + '_umap_' + str(i) + '.png')
    plt.close()
os.chdir('..')

pca = PCA(n_components=2)
embed_2d = pca.fit_transform(X)
plt.figure(figsize=(15, 14))
plt.scatter(embed_2d[:, 0], embed_2d[:, 1], c=colors, cmap='tab10', s=5)
plt.gca().set_aspect('equal', 'datalim')
plt.colorbar(boundaries=np.arange(n_types + 1) - 0.5).set_ticks(np.arange(n_types))
plt.title('PCA')
plt.savefig(filename + '_pca.png')
plt.close()

if not os.path.isdir(os.path.join(path, filename + '_pca_by_label')):
    os.mkdir(filename + '_pca_by_label')
os.chdir(filename + '_pca_by_label')
for i in range(len(label_dict)):
    new_color = [0 if j != label_dict[i] else 1 for j in colors]
    pca = PCA(n_components=2)
    embed_2d = pca.fit_transform(X)
    plt.figure(figsize=(15, 14))
    cmap = ListedColormap(['gray', 'blue'])
    alpha = [1 if c == 1 else 0.2 for c in new_color]
    plt.scatter(embed_2d[:, 0], embed_2d[:, 1], c=new_color, cmap=cmap, alpha=alpha)
    plt.gca().set_aspect('equal', 'datalim')
    plt.title('PCA')
    plt.savefig(filename + '_pca_' + str(i) + '.png')
    plt.close()
os.chdir('..')
