import torch
import torch.nn as nn
from torch_geometric.data import Batch, Data


class BridgeNodes(nn.Module):

    def __init__(self, feat_dim, group_num, node_num, threshold):
        super(BridgeNodes, self).__init__()

        self.feat_dim = feat_dim
        self.group_num = group_num
        self.node_num = node_num
        self.threshold = threshold

        self.node_group = nn.ParameterList([torch.nn.Parameter(torch.empty(node_num, feat_dim)) for _ in range(group_num)])
        for nodes in self.node_group:
            nn.init.kaiming_uniform_(nodes, nonlinearity='leaky_relu', mode='fan_in', a=0.01)

    def inner_update(self): # used for link bridge nodes
        bridge_graphs = []
        for i, nodes in enumerate(self.node_group):
            node_dot = torch.mm(nodes, torch.transpose(nodes, 0, 1))
            node_sim = torch.sigmoid(node_dot)

            inner_adj = torch.where(node_sim < self.threshold, 0, node_sim)
            edge_index = inner_adj.nonzero().t().contiguous()

            bridge_graphs.append(Data(x=nodes, edge_index=edge_index, y=torch.tensor([i]).long()))

        bridge_graphs_batch = Batch.from_data_list(bridge_graphs)

        return bridge_graphs_batch
