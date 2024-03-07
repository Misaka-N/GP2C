import torch
import wandb
from tqdm import tqdm
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
from models.GCN import GCN
from models.bridge_nodes import BridgeNodes
from utils.tools import cosine_similarity, EarlyStopping, set_random


class ContrastiveLearning(nn.Module):

    def __init__(self, GNN, output_dim, temperature, loss_bias):
        super(ContrastiveLearning, self).__init__()
        self.bias = 1e-4 # used for loss calculation
        self.T = temperature
        self.GNN = GNN
        self.projector = nn.Sequential(nn.Linear(output_dim, output_dim),
                                       nn.PReLU(),
                                       nn.Linear(output_dim, output_dim))
        self.loss_bias = loss_bias
        
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
        loss = -torch.log(loss).mean() + self.loss_bias
        
        return loss


class Augmentation(nn.Module):

    def __init__(self, augment: list, ratio: list, batch_size: int):
        for item in augment:
            if item not in ['Subgraph', 'Anonymize', 'Drop', 'Perturb', 'Mask']:
                raise ValueError('Using an unsupported method: ' + item)
        self.augment1 = augment[0]
        self.augment2 = augment[1]
        self.ratio1 = ratio[0]
        self.ratio2 = ratio[1]        
        self.batch_size = batch_size

        print("---Graph views: [{}, {}] with ratio: [{}, {}]---".format(self.augment1, self.augment2, self.ratio1, self.ratio2))

    def get_augmented_graph(self, graph_list):
        if len(graph_list) % self.batch_size == 1:
            raise KeyError("Batch_size {} makes the last batch only contain 1 graph, which will trigger a zero bug.".format(self.batch_size))
        
        shuffle(graph_list)

        aug1_list = []
        for g in tqdm(graph_list, desc="Augmentation: " + self.augment1):
            view_g = graph_views(graph=g, aug=self.augment1, ratio=self.ratio1)
            view_g = Data(x=view_g.x, edge_index=view_g.edge_index)
            aug1_list.append(view_g)
        loader1 = DataLoader(aug1_list, batch_size=1, shuffle=False, num_workers=0) # Must set shuffle=false, otherwise the sim_matrix is wrong!

        aug2_list = []
        for g in tqdm(graph_list, desc="Augmentation: " + self.augment2):
            view_g = graph_views(graph=g, aug=self.augment2, ratio=self.ratio2)
            view_g = Data(x=view_g.x, edge_index=view_g.edge_index)
            aug2_list.append(view_g)
        loader2 = DataLoader(aug2_list, batch_size=1, shuffle=False, num_workers=0) # Must set shuffle=false, otherwise the sim_matrix is wrong!

        return loader1, loader2
        

def connect_graphs(graph_list, bridge_nodes):
    big_x = [bridge_nodes.x]
    big_edge_index = []

    node_offset = bridge_nodes.x.size(0)

    for graph in graph_list:
        num_graph_nodes = graph.x.size(0)
        num_bridge_nodes = bridge_nodes.x.size(0)

        src = torch.arange(node_offset, node_offset + num_graph_nodes).repeat_interleave(num_bridge_nodes)
        dst = torch.arange(0, num_bridge_nodes).repeat(num_graph_nodes)

        reverse_src, reverse_dst = dst, src
        edges = torch.cat([torch.stack([src, dst], dim=0), torch.stack([reverse_src, reverse_dst], dim=0)], dim=1)

        big_edge_index.append(edges)
        big_x.append(graph.x)

        node_offset += num_graph_nodes
    
    big_x = torch.cat(big_x, dim=0)
    big_edge_index = torch.cat(big_edge_index, dim=1)

    big_graph = Data(x=big_x, edge_index=big_edge_index)
    return big_graph


def get_composed_graphs(augment, subgraph_num, batch_size, loader1_list, loader2_list, bridge_graph):
    graphs_for_aug1 = []
    for subgraphs in tqdm(zip(*loader1_list), desc="Augmentation: " + augment[0]): # simultaneously get one graph of a certain augmentation from each datasets
        composed_graph = connect_graphs(subgraphs, bridge_graph)
        graphs_for_aug1.append(composed_graph)
    graphs_for_aug1 = DataLoader(graphs_for_aug1, batch_size=batch_size, shuffle=False, num_workers=0)

    graphs_for_aug2 = []
    for subgraphs in tqdm(zip(*loader2_list), desc="Augmentation: " + augment[1]): # simultaneously get one graph of a certain augmentation from each datasets
        composed_graph = connect_graphs(subgraphs, bridge_graph)
        graphs_for_aug2.append(composed_graph)
    graphs_for_aug2 = DataLoader(graphs_for_aug2, batch_size=batch_size, shuffle=False, num_workers=0)

    return graphs_for_aug1, graphs_for_aug2


def contrastive_train(model, loader1, loader2, optimizer):
    model.train()
    train_loss, total_step = 0, 0
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


def adjust_subgraphs(node_num, batch_size, loader1, loader2, bridge_nodes, threshold):
    for composed_graph1 in loader1:
        # delete all links connected with bridge nodes
        edge_index = composed_graph1.edge_index
        mask = (edge_index[0] >= node_num) & (edge_index[1] >= node_num)
        edge_index = edge_index[:, mask]
        
        # Connect bridge nodes and subgraph nodes
        composed_graph1.x[:node_num] = bridge_nodes

        sim_matrix_out = cosine_similarity(composed_graph1.x[node_num:], composed_graph1.x[:node_num])
        src_out, dst_out = torch.where(sim_matrix_out >= threshold)

        new_edges_out = torch.cat([torch.stack([src_out, dst_out], dim=0), torch.stack([dst_out, src_out], dim=0)], dim=1)

        # Connect bridge nodes and bridge nodes
        sim_matrix_in = cosine_similarity(composed_graph1.x[:node_num], composed_graph1.x[:node_num])
        src_in, dst_in = torch.where(sim_matrix_in >= threshold)


        new_edges_in = torch.cat([torch.stack([src_in, dst_in], dim=0), torch.stack([dst_in, src_in], dim=0)], dim=1)

        edge_index = torch.cat((edge_index, new_edges_out, new_edges_in), dim=1)
        composed_graph1.edge_index = edge_index


    for composed_graph2 in loader2:
        # delete all links connected with bridge nodes
        edge_index = composed_graph2.edge_index
        mask = (edge_index[0] >= node_num) & (edge_index[1] >= node_num)
        edge_index = edge_index[:, mask]
        
        # Connect bridge nodes and subgraph nodes
        composed_graph2.x[:node_num] = bridge_nodes

        sim_matrix_out = cosine_similarity(composed_graph2.x[node_num:], composed_graph2.x[:node_num])
        src_out, dst_out = torch.where(sim_matrix_out >= threshold)

        new_edges_out = torch.cat([torch.stack([src_out, dst_out], dim=0), torch.stack([dst_out, src_out], dim=0)], dim=1)

        # Connect bridge nodes and bridge nodes
        sim_matrix_in = cosine_similarity(composed_graph2.x[:node_num], composed_graph2.x[:node_num])
        src_in, dst_in = torch.where(sim_matrix_in >= threshold)

        new_edges_in = torch.cat([torch.stack([src_in, dst_in], dim=0), torch.stack([dst_in, src_in], dim=0)], dim=1)

        edge_index = torch.cat((edge_index, new_edges_out, new_edges_in), dim=1)
        composed_graph2.edge_index = edge_index

    return loader1, loader2


if __name__ == "__main__":
    args = get_pretrain_args()

    print("PyTorch version:", torch.__version__)

    if torch.cuda.is_available():
        print("CUDA is available")
        print("CUDA version:", torch.version.cuda)
        device = torch.device("cuda")
        set_random(args.seed, True)
    else:
        print("CUDA is not available")
        device = torch.device("cpu")
        set_random(args.seed, False)

    # Get pretrain datasets
    # TODO: Deal with Graph Tasks(Now only support node tasks)
    graph_list = []
    if args.task == 'node': # Node level tasks
        for dataset in args.dataset:
            print("---Downloading dataset: " + dataset + "---")
            data, dataname, _ = pretrain_dataloader(input_dim=args.input_dim, dataset=dataset)
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
    loader1_list, loader2_list = [], []
    augmentation = Augmentation(augment=args.augment, ratio=args.aug_ratio, batch_size=args.batch_size)
    for graphs, dataset in zip(graph_list, args.dataset):
        print("---Augmenting dataset: " + dataset + "---")
        loader1, loader2 = augmentation.get_augmented_graph(graphs)
        loader1_list.append(loader1)
        loader2_list.append(loader2)

    # Get composed graphs
    print("---Getting composed graphs---")
    bridge_nodes = BridgeNodes(feat_dim=args.node_dim, node_num=args.node_num, threshold=args.threshold)
    bridge_graph = bridge_nodes.inner_update()
    composed_graph1, composed_graph2 = get_composed_graphs(args.augment, args.subgraphs, args.batch_size, loader1_list, loader2_list, bridge_graph)

    # Pretrain GNN
    print("---Pretraining GNN---")
    # gnn = GIN(num_layers=args.gnn_layer, feat_dim=args.input_dim, hidden_dim=args.hidden_dim, output_dim=args.output_dim)
    gnn = GCN(gcn_layer_num=args.gnn_layer, input_dim=args.input_dim, hid_dim=args.hidden_dim, out_dim=args.output_dim)
    model = ContrastiveLearning(GNN=gnn, output_dim=args.output_dim, temperature=args.temperature, loss_bias=args.loss_bias)
    optimizer = optim.Adam(list(gnn.parameters())+list(bridge_nodes.parameters()), lr=args.lr, weight_decay=args.decay)
    early_stopper = EarlyStopping(path1=args.path, patience=args.patience, min_delta=0)
 
    # start a new wandb run to track this script
    wandb.init(
        # set the wandb project where this run will be logged
        project="l2s_pretrain",
        # track hyperparameters and run metadata
        config=args.__dict__
        )

    for epoch in range(args.max_epoches):
        train_loss = contrastive_train(model=model, loader1=composed_graph1, loader2=composed_graph2, optimizer=optimizer)
        wandb.log({"loss": train_loss}) # log the loss to wandb
        print("Epoch: {} | train_loss: {:.5}".format(epoch+1, train_loss))

        early_stopper(model, train_loss)
        if early_stopper.early_stop:
            print("Stopping training...")
            print("Best Score: ", early_stopper.best_score)
            break
        else:
            composed_graph1, composed_graph2 = adjust_subgraphs(args.node_num, args.batch_size, composed_graph1, composed_graph2, bridge_nodes.node_group, args.threshold)

