import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class GCN(torch.nn.Module):
    def __init__(self, input_dim, hid_dim, out_dim, gcn_layer_num=2, pool=None):
        super().__init__()

        if gcn_layer_num < 2:
            raise ValueError('GNN layer_num should >=2 but you set {}'.format(gcn_layer_num))
        elif gcn_layer_num == 2:
            self.conv_layers = torch.nn.ModuleList([GCNConv(input_dim, hid_dim), GCNConv(hid_dim, out_dim)])
            # self.bn_layers = nn.ModuleList([nn.BatchNorm1d(hid_dim)])
        else:
            layers = [GCNConv(input_dim, hid_dim)]

            # bn_layers = [nn.BatchNorm1d(hid_dim)]
            for _ in range(gcn_layer_num - 2):
                layers.append(GCNConv(hid_dim, hid_dim))
                # bn_layers.append(nn.BatchNorm1d(hid_dim))
            layers.append(GCNConv(hid_dim, out_dim))
            self.conv_layers = torch.nn.ModuleList(layers)
            # self.bn_layers = nn.ModuleList(bn_layers)

    def forward(self, x, edge_index, batch):
        for i, conv in enumerate(self.conv_layers[:-1]):
            x = conv(x, edge_index)

            # if i < len(self.bn_layers):
                # x = self.bn_layers[i](x)
            x = act(x)
            x = F.dropout(x, training=self.training, p=0.7)

        node_emb = self.conv_layers[-1](x, edge_index)
        return node_emb
    
    def forward_in(self, x, edge_index, batch):
        x = self.conv_layers[0](x, edge_index)
        x = self.bn_layers[0](x)
        x = act(x)
        node_emb = F.dropout(x, training=self.training,p=0.7)

        return node_emb
    
    def forward_hid(self, x, edge_index, batch):
        for i, conv in enumerate(self.conv_layers[1:-2]):
            x = conv(x, edge_index)

            if i < len(self.bn_layers):
                x = self.bn_layers[i](x)
            x = act(x)
            x = F.dropout(x, training=self.training,p=0.7)

        node_emb = self.conv_layers[-1](x, edge_index)

        return node_emb
    
    def forward_out(self, x, edge_index, batch):
        x = self.conv_layers[-1](x, edge_index)
        x = act(x)
        node_emb = F.dropout(x, training=self.training,p=0.7)

        return node_emb


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
