import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool


class GCN(torch.nn.Module):
    def __init__(self, input_dim, hid_dim, out_dim, gcn_layer_num=2, pool=None):
        super().__init__()

        if gcn_layer_num < 2:
            raise ValueError('GNN layer_num should >=2 but you set {}'.format(gcn_layer_num))
        elif gcn_layer_num == 2:
            self.conv_layers = torch.nn.ModuleList([GCNConv(input_dim, hid_dim), GCNConv(hid_dim, out_dim)])
        else:
            layers = [GCNConv(input_dim, hid_dim)]
            for i in range(gcn_layer_num - 2):
                layers.append(GCNConv(hid_dim, hid_dim))
            layers.append(GCNConv(hid_dim, out_dim))
            self.conv_layers = torch.nn.ModuleList(layers)

        if pool is None:
            self.pool = global_mean_pool
        else:
            self.pool = pool

    def forward(self, x, edge_index, batch):
        for conv in self.conv_layers[0:-1]:
            x = conv(x, edge_index)
            x = act(x)
            x = F.dropout(x, training=self.training)

        node_emb = self.conv_layers[-1](x, edge_index)
        graph_emb = self.pool(node_emb, batch.long())
        return graph_emb


def act(x=None, act_type='leakyrelu'):
    if act_type == 'leakyrelu':
        if x is None:
            return torch.nn.LeakyReLU()
        else:
            return F.leaky_relu(x)
    elif act_type == 'tanh':
        if x is None:
            return torch.nn.Tanh()
        else:
            return torch.tanh(x)
