import os
import numpy as np
import random
import torch
from copy import deepcopy
from random import shuffle
from torch_geometric.data import Data
from torch_geometric.utils import subgraph, k_hop_subgraph
import pickle as pk
from torch_geometric.utils import to_undirected
from torch_geometric.loader.cluster import ClusterData
from torch import nn, optim

import torch.nn.functional as F


def subgraph_extraction(graph, radio):
    edge_index = graph.edge_index.numpy()
    degrees = np.bincount(edge_index[0], minlength=graph.x.size(0)) + np.bincount(edge_index[1], minlength=graph.x.size(0))
    
    node_num = graph.x.size(0)
    subgraph_node_num = int(node_num * radio)
    
    idx_sorted_by_degree = torch.argsort(torch.tensor(degrees, dtype=torch.float), descending=True)
    idx_subgraph = idx_sorted_by_degree[:subgraph_node_num]
    
    mask = torch.tensor(np.in1d(edge_index[0], idx_subgraph.numpy()) & np.in1d(edge_index[1], idx_subgraph.numpy()), dtype=torch.bool)
    edges_subgraph = torch.tensor(edge_index[:, mask.numpy()], dtype=torch.long)
    
    subgraph = graph.clone()
    subgraph.x = graph.x[idx_subgraph]
    subgraph.edge_index = edges_subgraph
    
    idx_mapping = {idx.item(): i for i, idx in enumerate(idx_subgraph)}
    for i in range(subgraph.edge_index.size(1)):
        subgraph.edge_index[0, i] = idx_mapping[subgraph.edge_index[0, i].item()]
        subgraph.edge_index[1, i] = idx_mapping[subgraph.edge_index[1, i].item()]
    
    return subgraph


def anonymization(graph, radio):
    node_num, _ = graph.x.size()
    idx_shuffle = np.random.permutation(node_num)
    graph.x = graph.x[idx_shuffle]
    
    return graph


def node_dropping(graph, radio):
    node_num, _ = graph.x.size()
    _, edge_num = graph.edge_index.size()
    drop_num = int(node_num * radio)

    idx_perm = np.random.permutation(node_num) # Randomly arrange node index

    idx_drop = idx_perm[:drop_num] # dropped nodes
    idx_nondrop = idx_perm[drop_num:] # rest nodes
    idx_nondrop.sort()
    idx_dict = {idx_nondrop[n]: n for n in list(range(idx_nondrop.shape[0]))} # old_idx -> new_idx

    edge_index = graph.edge_index.numpy()

    edge_index = [[idx_dict[edge_index[0, n]], idx_dict[edge_index[1, n]]] for n in range(edge_num) 
                  if (not edge_index[0, n] in idx_drop) and (not edge_index[1, n] in idx_drop)] # drop edges that connected with dropped nodes
    
    graph.edge_index = torch.tensor(edge_index).transpose_(0, 1)
    graph.x = graph.x[idx_nondrop]

    return graph


def edge_perturbation(graph, radio):
    _, edge_num = graph.edge_index.size()
    perturb_num = int(edge_num * radio)

    idx_delete = np.random.choice(edge_num, (edge_num - perturb_num), replace=False)
    graph.edge_index = graph.edge_index[:, idx_delete]

    return graph


def attribute_masking(graph, radio):
    node_num, _ = graph.x.size()
    mask_num = int(node_num * radio)
    
    idx_perm = np.random.permutation(node_num)[:mask_num]
    mask = torch.ones_like(graph.x)
    mask[idx_perm] = 0 # Another way is to use a certain value, like a 'mean' value of the graph
    
    graph.x = graph.x * mask
    
    return graph


def graph_views(graph, aug, radio):
    if aug == 'Subgraph':
        aug_graph = subgraph_extraction(graph, radio)
    elif aug == 'Anonymize':
        aug_graph = anonymization(graph, radio)
    elif aug == 'Drop':
        aug_graph = node_dropping(graph, radio)
    elif aug == 'Perturb':
        aug_graph = edge_perturbation(graph, radio)
    elif aug == 'Mask':
        aug_graph = attribute_masking(graph, radio)

    return aug_graph