import torch
import torch.nn as nn
from torch_geometric.data import Batch, Data


class BridgeNodes(nn.Module):

    def __init__(self, feat_dim, node_num, threshold):
        super(BridgeNodes, self).__init__()

        self.feat_dim = feat_dim
        self.node_num = node_num
        self.threshold = threshold

        self.node_group = nn.Parameter(torch.randn(node_num, feat_dim))
        nn.init.kaiming_uniform_(self.node_group, nonlinearity='leaky_relu', mode='fan_in', a=0.01)

    def inner_update(self): # used for link bridge nodes
        node_dot = torch.mm(self.node_group, torch.transpose(self.node_group, 0, 1))
        node_sim = torch.sigmoid(node_dot)

        inner_adj = torch.where(node_sim < self.threshold, 0, node_sim)
        edge_index = inner_adj.nonzero().t().contiguous()

        bridge_graph = Data(x=self.node_group, edge_index=edge_index)

        return bridge_graph
