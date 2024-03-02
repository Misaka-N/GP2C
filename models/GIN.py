import torch
from torch.nn import Sequential, Linear, ReLU, BatchNorm1d
from torch_geometric.nn import GINConv, global_add_pool
from torch_geometric.data import DataLoader

class GIN(torch.nn.Module):
    def __init__(self, num_layers, feat_dim, hidden_dim, out_dim):
        super(GIN, self).__init__()
        self.convs = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()

        # First Layer
        nn1 = Sequential(Linear(feat_dim, hidden_dim), ReLU(), Linear(hidden_dim, hidden_dim))
        self.convs.append(GINConv(nn1))
        self.bns.append(BatchNorm1d(hidden_dim))
        
        # L-1 Layers
        for _ in range(num_layers - 1):
            nn = Sequential(Linear(hidden_dim, hidden_dim), ReLU(), Linear(hidden_dim, hidden_dim))
            self.convs.append(GINConv(nn))
            self.bns.append(BatchNorm1d(hidden_dim))
        
        # Full Connected Layer
        self.fc = torch.nn.Sequential(Linear(hidden_dim, hidden_dim),
                                      ReLU(),
                                      Linear(hidden_dim, out_dim))
    
    def forward(self, x, edge_index, batch):
        for conv, bn in zip(self.convs, self.bns):
            x = conv(x, edge_index)
            x = bn(x)
            x = torch.relu(x)
        
        x = global_add_pool(x, batch)
        x = self.fc(x)
        
        return x
