import os

import numpy as np
import pandas as pd
import scanpy as sc
import torch
from torch_geometric.loader import DataLoader


device = torch.device('cpu')
model = torch.load('model.pth')
model.to(device)
data = sc.read_text('scRNA-data/Chutype/logChumarker.txt')
data = data.transpose()
cell_type = pd.read_csv('scRNA-data/Chutype/chutypectname.txt', sep=' ')  # The observation labeled with cell types.
cell_type.index = data.obs.index
data.obs = cell_type
values = data.obs['cell_type'].value_counts()
n_types = values.shape[0]
datas = []
os.chdir('Datas_Chutype')
filelist = os.listdir(os.getcwd())

for i in range(len(filelist)):
    data = torch.load(filelist[i])
    data.to(device)
    datas.append(data)
os.chdir('..')
new_values = pd.read_csv('new_values.csv')
loader = DataLoader(datas, batch_size=1024, shuffle=False)
n_cells = len(datas)
hidden_layer = 128
embeddings = []

model.eval()
for data in loader:
    _, embedding = model(data.x.to(device), data.edge_index.to(device), data.batch.to(device))
    embedding = embedding.cpu().detach().numpy()
    embeddings.append(embedding)

embeddings = np.concatenate(embeddings, axis=0)

np.savetxt('embedding_by_try_ano_CSN.txt', embeddings)
