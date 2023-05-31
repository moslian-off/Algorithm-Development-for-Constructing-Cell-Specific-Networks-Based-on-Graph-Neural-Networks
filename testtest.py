import pandas as pd
import scanpy as sc

data = sc.read_text('scRNA-data/Chutype/logChumarker.txt')
data = data.transpose()
cell_type = pd.read_csv('scRNA-data/Chutype/chutypectname.txt', sep=' ')  # The observation labeled with cell types.
cell_type.index = data.obs.index
data.obs = cell_type

genes_name = ['PECAM1', 'POU5F1', 'ZFP42', 'NANOG', 'PRDM14', 'MT1X', 'ZFHX4']
data_subset = data[:, genes_name]

sc.pl.heatmap(data_subset, data_subset.var.index, groupby="cell_type", dendrogram=False, swap_axes=True,
              show_gene_labels=True, cmap='Wistia', figsize=(8, 6))
