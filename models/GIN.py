import torch
from torch.nn import Sequential, Linear, ReLU, BatchNorm1d, LayerNorm
from torch_geometric.nn import GINConv, global_add_pool
from torch_geometric.data import DataLoader

class GIN(torch.nn.Module):
    def __init__(self, num_layers, feat_dim, hidden_dim, output_dim):
        super(GIN, self).__init__()
        self.convs = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()
        self.norm = torch.nn.ModuleList()

        # First Layer
        nn1 = Sequential(Linear(feat_dim, hidden_dim), ReLU(), Linear(hidden_dim, hidden_dim))
        self.convs.append(GINConv(nn1))
        self.bns.append(BatchNorm1d(hidden_dim))
        self.norm.append(LayerNorm(hidden_dim))
        
        # L-1 Layers
        for _ in range(num_layers - 1):
            nn = Sequential(Linear(hidden_dim, hidden_dim), ReLU(), Linear(hidden_dim, hidden_dim))
            self.convs.append(GINConv(nn))
            self.bns.append(BatchNorm1d(hidden_dim))
            self.norm.append(LayerNorm(hidden_dim))
        
        # Full Connected Layer
        self.fc = torch.nn.Sequential(Linear(hidden_dim, hidden_dim),
                                      ReLU(),
                                      Linear(hidden_dim, output_dim),
                                      ReLU())
        self.norm.append(LayerNorm(output_dim))
    
    def forward(self, x, edge_index, batch, prompt=None, layers=-1):
        if prompt != None:
            if layers == -1: # shallow prompt
                for conv, norm in zip(self.convs, self.norm):
                    x = conv(x, edge_index)
                    x = torch.relu(x)
                    x = norm(x)
                sim_matrix = torch.matmul(x, prompt.t())
                x = x + torch.matmul(sim_matrix, prompt)
                x = torch.relu(x)

            else: # deep prompt
                for conv, norm, prompt_layer in zip(self.convs, self.norm, prompt):
                    x = conv(x, edge_index)
                    sim_matrix = torch.matmul(x, prompt_layer.t())
                    x = x + torch.matmul(sim_matrix, prompt_layer)
                    x = torch.relu(x)
                    x = norm(x)
        else:
            for conv, bn, norm in zip(self.convs, self.bns, self.norm):
                x = conv(x, edge_index)
                x = bn(x)
                x = torch.relu(x)
                x = norm(x)
        
        x = global_add_pool(x, batch)
        x = self.fc(x)
        x = self.norm[-1](x)

        return x
