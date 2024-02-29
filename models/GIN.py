import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn import GINConv

class ApplyNodeFunc(nn.Module):
    """
    Applying a multi-layer perceptron (MLP) to each node
    """
    def __init__(self, mlp):
        super(ApplyNodeFunc, self).__init__()
        self.mlp = mlp

    def forward(self, h):
        h = self.mlp(h)
        return h

class GIN(nn.Module):
    def __init__(self, num_layers, num_mlp_layers, input_dim, hidden_dim, output_dim):
        super(GIN, self).__init__()
        self.layers = nn.ModuleList()
        
        # FIrst GIN, dimension is input_dim*hidden_dim
        mlp = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim))
        self.layers.append(GINConv(ApplyNodeFunc(mlp), 'sum'))
        
        # Other GINs are hidden_dim*hidden_dim
        for layer in range(num_layers - 1):
            mlp = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim))
            self.layers.append(GINConv(ApplyNodeFunc(mlp), 'sum'))
        
        # Output layer
        self.linear = nn.Linear(hidden_dim, output_dim)

    def forward(self, g, features):
        h = features
        for layer in self.layers:
            h = layer(g, h)
        g.ndata['h'] = h
        hg = dgl.mean_nodes(g, 'h')
        return self.linear(hg)

