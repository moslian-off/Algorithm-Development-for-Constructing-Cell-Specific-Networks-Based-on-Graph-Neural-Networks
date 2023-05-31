import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import torch
import torch.nn.functional as f
from sklearn.metrics import roc_auc_score
from torch_geometric.loader import DataLoader

from ano_model import GCNNet
from sample import sample

data = sc.read_text('scRNA-data/Chutype/logChumarker.txt')
data = data.transpose()
cell_type = pd.read_csv('scRNA-data/Chutype/chutypectname.txt', sep=' ')  # The observation labeled with cell types.
cell_type.index = data.obs.index
data.obs = cell_type
values = data.obs['cell_type'].value_counts()
n_types = values.shape[0]
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
datas = []
os.chdir('Datas_Chutype')
filelist = os.listdir(os.getcwd())
for i in range(len(filelist)):
    data = torch.load(filelist[i])
    data.to(device)
    datas.append(data)
os.chdir('..')
new_values = pd.read_csv('new_values.csv')
train_dataset, test_dataset = sample(datas, new_values)

train_loader = DataLoader(train_dataset, batch_size=1024, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1024, shuffle=False)

model = GCNNet(2, 128, n_types).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-5, weight_decay=1e-6)
criterion = torch.nn.CrossEntropyLoss()
train_probs = []
test_prob = []


def train():
    global train_loader, n_types
    model.train()
    losses = []
    for data in train_loader:
        optimizer.zero_grad()
        out, _ = model(data.x.to(device), data.edge_index.to(device), data.batch.to(device))  # 将数据移动到GPU
        loss = criterion(out, data.y.to(device))
        losses.append(loss)
        loss.backward()
        optimizer.step()
    ave_loss = sum(losses) * 1000 / len(train_loader.dataset)
    return ave_loss


def test(loader):
    model.eval()
    correct = 0
    preds_prob = []
    labels = []
    for data in loader:
        out, _ = model(data.x.to(device), data.edge_index.to(device), data.batch.to(device))
        pred_prob = f.softmax(out, dim=1)
        preds_prob.append(pred_prob.detach().cpu().numpy())
        label = data.y.cpu().numpy()
        labels.append(label)
        pred = out.argmax(dim=1)
        correct += int((pred == data.y).sum())
        preds_prob = np.concatenate(preds_prob)
        labels = np.concatenate(labels)
    accuracy = correct / len(loader.dataset)
    aroc = roc_auc_score(labels, preds_prob, multi_class='ovo')
    return accuracy, aroc


epochs = 80000
losses = np.zeros(epochs)
train_acces = np.zeros(epochs)
test_acces = np.zeros(epochs)
test_arocs = np.zeros(epochs)
train_arocs = np.zeros(epochs)
for epoch in range(epochs):
    ave_loss = train()
    train_acc, train_auroc = test(train_loader)
    test_acc, test_aroc = test(test_loader)
    losses[epoch] = ave_loss
    train_acces[epoch] = train_acc
    test_acces[epoch] = test_acc
    test_arocs[epoch] = test_aroc
    train_arocs[epoch] = train_auroc
    print(f'Epoch: {epoch:03d}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}, '
          f'Train AUROC: {train_auroc:.4f}, Test AUROC: {test_aroc:.4f}, Loss: {ave_loss:.6f}')

torch.save(model, 'model.pth')
plt.plot(losses)
plt.title('Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.savefig('Training_loss.png')
plt.close()

plt.plot(train_acces, label='Train')
plt.plot(test_acces, label='Test')
plt.title('Training and Testing Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig('acc.png')
plt.close()

plt.plot(train_arocs, label='Train')
plt.plot(test_arocs, label='Test')
plt.title('Training and Testing AROCs')
plt.xlabel('Epoch')
plt.ylabel('AUROC')
plt.legend()
plt.savefig('AUROC.png')
plt.close()
