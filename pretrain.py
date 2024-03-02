import tqdm
import torch
import random
import argparse
import torch.nn as nn
import torch.optim as optim
from random import shuffle
from utils.args import get_pretrain_args
from utils.dataloader import pretrain_dataloader
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected
from torch_geometric.loader.cluster import ClusterData
from utils.augment import graph_views
from models.GIN import GIN
from models.bridge_nodes import BridgeNodes
from utils.tools import cosine_similarity, EarlyStopping


class ContrastiveLearning(nn.Module):

    def __init__(self, GNN, out_dim, temperature):
        super(ContrastiveLearning, self).__init__()
        self.bias = 1e-4 # used for loss calculation
        self.T = temperature
        self.GNN = GNN
        self.projector = nn.Sequential(nn.Linear(out_dim, out_dim),
                                       nn.PReLU(),
                                       nn.Linear(out_dim, out_dim))
        
    def forward_cl(self, x, edge_index, batch):
        x = self.GNN(x, edge_index, batch)
        x = self.projector(x)

        return x
    
    def loss_cl(self, x1, x2):
        batch_size, _ = x1.size()
        sim_matrix = cosine_similarity(x1, x2) # similarity calculation
        sim_matrix = torch.exp(sim_matrix / self.T) # using temperature to scale results
        pos_sim = sim_matrix[range(batch_size), range(batch_size)] # calculate for positive samples(in diagonal of sim_matrix )
        loss = pos_sim / ((sim_matrix.sum(dim=1) - pos_sim) + self.bias)
        loss = -torch.log(loss).mean()


class Augmentation(nn.Module):

    def __init__(self, augment: list, radio: list, batch_size: int):
        for item in augment:
            if item not in ['Subgraph', 'Anonymize', 'Drop', 'Perturb', 'Mask']:
                raise ValueError('Using an unsupported method: ' + item)
        self.augment = augment
        self.radio = radio
        self.batch_size = batch_size

        print("---Graph views: {} with radio: {}---".format(self.augment, self.radio))

    def get_augmented_graph(self, graph_list):
        if len(graph_list) % self.batch_size == 1:
            raise KeyError("Batch_size {} makes the last batch only contain 1 graph, which will trigger a zero bug.".format(batch_size))
        
        view_list, loader = [], []
        shuffle(graph_list)
        for aug, radio in zip(self.augment, self.radio):
            temp_list = []
            for g in tqdm(graph_list, desc="Augmentation: " + aug):
                view_g = graph_views(graph=g, aug=aug, radio=radio)
                view_g = Data(x=view_g.x, edge_index=view_g.edge_index)
                temp_list.append(view_g)
            view_list.append(temp_list)

        for view in view_list:
            temp_loader = DataLoader(view, batch_size=1, shuffle=False, num_workers=1) # Must set shuffle=false, otherwise the sim_matrix is wrong!
            loader.append(temp_loader)

        return loader # [aug_num, batch_num, batch_size]
        

def connect_graphs(graph_list, bridge_nodes, threshold):
    big_x = [bridge_nodes.x]
    big_edge_index = []

    node_offset = bridge_nodes.x.size(0)

    for graph in graph_list:
        sim_matrix = cosine_similarity(graph.x, bridge_nodes.x)
        src, dst = torch.where(sim_matrix >= threshold)
        
        src, dst = src + node_offset, dst
        reverse_src, reverse_dst = dst, src # undirected edge
        
        edges = torch.cat([torch.stack([src, dst], dim=0), torch.stack([reverse_src, reverse_dst], dim=0)], dim=1)
        
        big_edge_index.append(edges)
        big_x.append(graph.x)
        
        node_offset += graph.x.size(0)
    
    big_x = torch.cat(big_x, dim=0)
    big_edge_index = torch.cat(big_edge_index, dim=1)

    big_graph = Data(x=big_x, edge_index=big_edge_index)
    return big_graph

def contrastive_train(model, loaders, optimizer):
    model.train()
    train_loss, total_step = 0, 0
    for i, loader1 in enumerate(loaders):
        for _, loader2 in enumerate(loaders[i+1:], start=i+1):
            for _, batch in enumerate(zip(loader1, loader2)):
                batch1, batch2 = batch
                optimizer.zero_grad()
                x1 = model.forward_cl(batch1.x, batch1.edge_index, batch1.batch)
                x2 = model.forward_cl(batch2.x, batch2.edge_index, batch2.batch)
                loss = model.loss_cl(x1, x2)

                loss.backward()
                optimizer.step()

                train_loss += float(loss.detach().cpu().item())
                total_step = total_step + 1

    return train_loss / total_step

if __name__ == "__main__":
    print("PyTorch version:", torch.__version__)

    if torch.cuda.is_available():
        print("CUDA is available")
        print("CUDA version:", torch.version.cuda)
        device = torch.device("cuda")
    else:
        print("CUDA is not available")
        device = torch.device("cpu")

    args = get_pretrain_args()

    # Get pretrain datasets
    # TODO: Deal with Graph Tasks(Now only support node tasks)
    graph_list = []
    if args.task == 'node': # Node level tasks
        for dataset in args.dataset:
            print("---Downloading dataset: " + dataset + "---")
            data, dataname = pretrain_dataloader(input_dim=args.input_dim, dataset=dataset)
            print("---Getting subgraphs of dataset: " + dataset + "---")
            x = data.x.detach()
            edge_index = data.edge_index
            edge_index = to_undirected(edge_index)
            data = Data(x=x, edge_index=edge_index)
            graphs = list(ClusterData(data=data, num_parts=args.subgraphs, save_dir='data/{}/'.format(dataname)))
            graph_list.append(graphs)
    else:
        pass

    # Get augmented graphs
    loader_list = []
    augmentation = Augmentation(augment=args.augment, radio=args.aug_radio, batch_size=args.batch_size)
    for graphs, dataset in zip(graph_list, args.dataset):
        print("---Augmenting dataset: " + dataset + "---")
        loader = augmentation.get_augmented_graph(graphs)
        loader_list.append(loader)

    # Get composed graphs
    bridge_nodes = BridgeNodes(feat_dim=args.node_dim, group_num=args.node_group, node_num=args.node_num, threshold=args.threshold)
    bridge_graphs_batch = DataLoader(bridge_nodes, batch_size=1, shuffle=True, num_workers=1)
    composed_graphs = []
    for augs in zip(*loader_list): # get aug_X simultaneously from all datasets to compose subgraphs
        graphs_for_one_aug = []
        for i, batch in enumerate(zip(*augs, bridge_graphs_batch)): # simultaneously get one graph of a certain augmentation from each datasets
            *subgraphs, bridge_nodes = batch
            composed_graph = connect_graphs(subgraphs, bridge_nodes, args.threshold)
            graphs_for_one_aug.append(composed_graph)
        graphs_for_one_aug = DataLoader(graphs_for_one_aug, batch_size=args.batch_size, shuffle=False, num_workers=1)
    composed_graphs.append() 

    # Pretrain GNN
    gnn = GIN(num_layers=args.gnn_layer, feat_dim=args.input_dim, hidden_dim=args.hidden_dim, out_dim=args.out_dim)
    model = ContrastiveLearning(GNN=gnn, out_dim=args.out_dim, temperature=args.temperature)
    optimizer = optim.Adam(gnn.parameters(), lr=args.lr, weight_decay=args.decay)
    early_stopper = EarlyStopping(id=args.id, datasets=args.datasets, methods=args.methods, gnn_type='GIN', patience=args.patience, min_delta=0)
 
    for epoch in range(args.max_epoches):
        train_loss = contrastive_train(model=model, loaders=composed_graphs, optimizer=optimizer)
        print("Epoch: {} | train_loss: {:.5}".format(epoch+1, train_loss))

        early_stopper(train_loss)
        if early_stopper.early_stop:
            print("Stopping training...")
            break

