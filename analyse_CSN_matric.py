import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import umap
from sklearn import metrics
from sklearn.cluster import KMeans


filename = 'scRNA_matrix'
data = sc.read_text('scRNA-data/Chutype/logChumarker.txt')
data = data.transpose()
cell_type = pd.read_csv('scRNA-data/Chutype/chutypectname.txt', sep=' ')  # The observation labeled with cell types.
cell_type.index = data.obs.index
data.obs = cell_type

values = data.obs['cell_type'].value_counts()
n_types = values.shape[0]
labels = np.array(data.obs['cell_type'])
cbr_label = values.index.tolist()
label_dict = {'H1': 0, 'NPC': 1, 'H9': 2, 'HFF': 3, 'DEC': 4, 'EC': 5, 'TB': 6}
colors = np.zeros(data.n_obs)

for i in range(data.n_obs):
    colors[i] = label_dict[data.obs.iloc[i]['cell_type']]

Score_dic = {}

reducer = umap.UMAP(random_state=42, n_components=2)
X_umap = reducer.fit_transform(data.X)
plt.figure(figsize=(15, 14))
plt.scatter(X_umap[:, 0], X_umap[:, 1], c=colors, cmap='tab10', s=5)
plt.gca().set_aspect('equal', 'datalim')
cbr = plt.colorbar(boundaries=np.arange(n_types + 1) - 0.5)
cbr.set_ticks(np.arange(n_types))
cbr.set_ticklabels(cbr_label)
plt.title('UMAP')
plt.savefig('umap_from_scRNA_matrix.png')
plt.close()

km_umap = KMeans(n_clusters=n_types, random_state=42)
umap_labels = km_umap.fit_predict(X_umap)
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
Score.to_csv('Score_from_scRNA_matrix.csv', index=True)
