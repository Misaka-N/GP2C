import torch
import torch.nn as nn
from torch.nn import Sequential, Linear, ReLU, BatchNorm1d
from torch_geometric.nn import GINConv, global_add_pool
from torch_geometric.data import DataLoader

class GIN(torch.nn.Module):
    def __init__(self, num_layers, feat_dim, hidden_dim, output_dim):
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
                                      Linear(hidden_dim, output_dim))
    
    def forward(self, x, edge_index, batch, prompt=None):
        if prompt != None:
            if isinstance(prompt, nn.Parameter): # shallow prompt
                for conv, bn in zip(self.convs, self.bns):
                    x = conv(x, edge_index)
                    x = bn(x)
                    x = torch.relu(x)
                sim_matrix = torch.matmul(x, prompt.t())
                x = x + torch.matmul(sim_matrix, prompt)
                x = bn(x)
                x = torch.relu(x)

            else: # deep prompt
                for conv, bn, prompt_layer in zip(self.convs, self.bns, prompt):
                    x = conv(x, edge_index)
                    sim_matrix = torch.matmul(x, prompt_layer.t())
                    x = x + torch.matmul(sim_matrix, prompt_layer)
                    x = bn(x)
                    x = torch.relu(x)
        else:
            for conv, bn in zip(self.convs, self.bns):
                x = conv(x, edge_index)
                x = bn(x)
                x = torch.relu(x)
        
        x = global_add_pool(x, batch)
        x = self.fc(x)
        
        return x
