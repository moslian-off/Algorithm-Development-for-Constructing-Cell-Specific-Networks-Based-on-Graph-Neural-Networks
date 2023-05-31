import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch_geometric.nn import GCNConv, Linear
from torch_geometric.nn import global_mean_pool


class GCNNet(torch.nn.Module):
    def __init__(self, in_channels, hidden_layer, out_channels):
        super(GCNNet, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_layer)
        self.conv2 = GCNConv(hidden_layer, hidden_layer)
        self.conv3 = GCNConv(hidden_layer, hidden_layer)
        self.conv4 = GCNConv(hidden_layer, hidden_layer)
        self.conv5 = GCNConv(hidden_layer, hidden_layer)
        self.lin = Linear(hidden_layer, out_channels)

    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index)
        x = x.tanh()
        x = self.conv2(x, edge_index)
        x = x.tanh()
        x = self.conv3(x, edge_index)
        x = x.tanh()
        x = self.conv4(x, edge_index)
        x = x.tanh()
        x = self.conv5(x, edge_index)
        x = x.tanh()
        embedding = global_mean_pool(x, batch)  # [batch_size, hidden_channels]
        x = self.lin(embedding)
        return x, embedding


class GATNet(torch.nn.Module):
    def __init__(self, in_channels, hidden_layer, out_channels, heads=1):
        super(GATNet, self).__init__()
        self.conv1 = GATConv(in_channels, hidden_layer, heads=heads)
        self.conv2 = GATConv(hidden_layer, hidden_layer, heads=heads)
        self.conv3 = GATConv(hidden_layer, hidden_layer, heads=heads)
        self.lin = Linear(hidden_layer, out_channels)

    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = self.conv3(x, edge_index)
        x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]
        x = self.lin(x)
        return x
