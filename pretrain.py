import torch
import wandb
import numpy as np
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from random import shuffle
from utils.args import get_pretrain_args
from utils.dataloader import pretrain_dataloader
from torch_geometric.nn import global_mean_pool
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected, add_self_loops
from torch_geometric.loader.cluster import ClusterData
from utils.augment import graph_views
from models.GCN import GCN
from utils.tools import cosine_similarity, EarlyStopping, set_random, clone_module, update_module
import torch.nn.functional as F
from itertools import combinations
import random
from torch.autograd import grad
import traceback


import os
os.environ["WANDB_MODE"] = "offline"
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"


class ContrastiveLearning(nn.Module):

    def __init__(self, GNN, projector, temperature, loss_bias, first_order=False, allow_unused=None, allow_nograd=False):
        super(ContrastiveLearning, self).__init__()
        self.bias = 1e-4 # used for loss calculation
        self.T = temperature
        self.GNN = GNN
        self.projector = projector
        self.loss_bias = loss_bias
        self.pool = global_mean_pool

        self.first_order = first_order
        self.allow_nograd = allow_nograd
        if allow_unused is None:
            allow_unused = allow_nograd
        self.allow_unused = allow_unused
        
    def forward_cl(self, x, edge_index, batch):
        x = self.GNN(x, edge_index, batch)
        x = self.pool(x, batch.long())
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
    
    def meta_update(self, module, lr, grads=None):
        if grads is not None:
            params = list(module.parameters())
            if not len(grads) == len(list(params)):
                msg = 'WARNING:maml_update(): Parameters and gradients have different length. ('
                msg += str(len(params)) + ' vs ' + str(len(grads)) + ')'
                print(msg)
            for p, g in zip(params, grads):
                if g is not None:
                    p.update = - lr * g
        return update_module(module)

    def adapt(self, loss, first_order=None, allow_unused=None, allow_nograd=None):
        if first_order is None:
            first_order = self.first_order
        if allow_unused is None:
            allow_unused = self.allow_unused
        if allow_nograd is None:
            allow_nograd = self.allow_nograd
        second_order = not first_order

        parameters = list(self.GNN.parameters()) + list(self.projector.parameters())

        gnn_param_count = sum(1 for _ in self.GNN.parameters())

        if allow_nograd:
            diff_params = [p for p in parameters if p.requires_grad]
            grad_params = grad(loss, diff_params, retain_graph=second_order, create_graph=second_order, allow_unused=allow_unused)
            gradients = []
            grad_counter = 0

            for param in parameters:
                if param.requires_grad:
                    gradient = grad_params[grad_counter]
                    grad_counter += 1
                else:
                    gradient = None
                gradients.append(gradient)
        else:
            try:
                gradients = grad(loss, parameters, retain_graph=second_order, create_graph=second_order, allow_unused=allow_unused)
            except RuntimeError:
                traceback.print_exc()
                print('Maybe try with allow_nograd=True and/or allow_unused=True ?')

        gnn_gradients = gradients[:gnn_param_count]
        projector_gradients = gradients[gnn_param_count:]

        self.GNN = self.meta_update(self.GNN, args.lr, gnn_gradients)
        self.projector = self.meta_update(self.projector, args.lr, projector_gradients)

    def clone(self, first_order=None, allow_unused=None, allow_nograd=None):
        if first_order is None:
            first_order = self.first_order
        if allow_unused is None:
            allow_unused = self.allow_unused
        if allow_nograd is None:
            allow_nograd = self.allow_nograd

        return ContrastiveLearning(clone_module(self.GNN), clone_module(self.projector), temperature=self.T, loss_bias=self.loss_bias, first_order=first_order, allow_unused=allow_unused, allow_nograd=allow_nograd)
    

class Augmentation(nn.Module):

    def __init__(self, augment: list, ratio: list, batch_size: int):
        for item in augment:
            if item not in ['Subgraph', 'Anonymize', 'Drop', 'Perturb', 'Mask', 'None']:
                raise ValueError('Using an unsupported method: ' + item)
        self.augments = augment
        self.ratios = ratio
        self.batch_size = batch_size

        print("---Graph views: {} with ratio: {} ---".format(self.augments, self.ratios))

    def get_augmented_graph(self, graph_list):
        if len(graph_list) % self.batch_size == 1:
            raise KeyError("Batch_size {} makes the last batch only contain 1 graph, which will trigger a zero bug.".format(self.batch_size))
        
        shuffle(graph_list)

        aug_dict = {}
        for aug, radio in zip(self.augments, self.ratios):
            aug_list = []
            for g in tqdm(graph_list, desc="Augmentation: " + aug):
                view_g = graph_views(graph=g, aug=aug, ratio=radio)
                view_g = Data(x=view_g.x, edge_index=view_g.edge_index)
                aug_list.append(view_g)
            aug_dict[aug] = aug_list

        return aug_dict


def contrastive_train(model, loader1, loader2):
    train_loss, total_step = 0, 0
    for _, batch in enumerate(zip(loader1, loader2)):
        batch1, batch2 = batch

        x1 = model.forward_cl(batch1.x.to(device), batch1.edge_index.to(device), batch1.batch.to(device))
        x2 = model.forward_cl(batch2.x.to(device), batch2.edge_index.to(device), batch2.batch.to(device))

        loss = model.loss_cl(x1, x2)

        train_loss += loss
        total_step = total_step + 1

    return train_loss / total_step


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
            graphs = list(ClusterData(data=data, num_parts=args.subgraphs, save_dir='../autodl-tmp/data/{}/'.format(dataname)))
            graph_list.append(graphs)
    elif args.task == 'graph':
        for dataset in args.dataset:
            print("---Downloading dataset: " + dataset + "---")
            graphs, dataname, _ = pretrain_dataloader(input_dim=args.input_dim, dataset=dataset)  
            graph_list.append(graphs)
            # quit()
            

    
    # Get augmented graphs
    aug_datasets = []
    augmentation = Augmentation(augment=args.augment, ratio=args.aug_ratio, batch_size=args.batch_size)   
    for graphs, dataset in zip(graph_list, args.dataset):
        print("---Augmenting dataset: " + dataset + "---")
        aug_graphs = augmentation.get_augmented_graph(graphs)
        aug_datasets.append(aug_graphs)

    # Get support tasks
    support_tasks = []
    for aug_dataset in aug_datasets:
        aug_dataset_list = [aug_dataset[aug] for aug in aug_dataset.keys()]
        task_graphs = list(combinations(aug_dataset_list, 2))
        support_tasks.extend(task_graphs)
    support_tasks = [(DataLoader(task[0], batch_size=args.batch_size, shuffle=False, num_workers=0), DataLoader(task[1], batch_size=args.batch_size, shuffle=False, num_workers=0)) for task in support_tasks]
    shuffle(support_tasks)

    # Pretrain setup
    print("---Pretraining GNN---")
    gnn = GCN(gcn_layer_num=args.gnn_layer, input_dim=args.input_dim, hid_dim=args.hidden_dim, out_dim=args.output_dim).to(device)
    projector = nn.Sequential(nn.Linear(args.output_dim, args.output_dim), nn.PReLU(), nn.Linear(args.output_dim, args.output_dim)).to(device)
    model = ContrastiveLearning(GNN=gnn, projector=projector, temperature=args.temperature, loss_bias=args.loss_bias).to(device)
    optimizer = optim.Adam(list(gnn.parameters()) + list(projector.parameters()), lr=args.lr, weight_decay=args.decay)
    early_stopper = EarlyStopping(path1=args.path, patience=args.patience, min_delta=0)
 
    wandb.init(
        project="l2s_pretrain",
        config=args.__dict__,
        name='+'.join(args.dataset) + " " + str(args.seed)
        )
    
    # Meta-Learning for pretraining gnn
    for epoch in range(args.max_epoches):
        meta_loss = 0.

        for task_id, support_task in enumerate(support_tasks):
            aug_loader1, aug_loader2 = support_task
            meta_model = model.clone()

            for step in range(args.adapt_step):
                adapt_loss = contrastive_train(model=meta_model, loader1=aug_loader1, loader2=aug_loader2)
                print("Epoch: {} | Task: {} | Step: {} | Adapt-Loss: {:.5f}".format(epoch, task_id, step, adapt_loss.item()))
                meta_model.adapt(adapt_loss)

            update_loss = contrastive_train(model=meta_model, loader1=aug_loader1, loader2=aug_loader2)
            print("Epoch: {} | Task: {} | Update-Loss: {:.5f}".format(epoch, task_id, update_loss.item()))
            meta_loss += update_loss
        
        meta_loss = meta_loss / len(support_tasks)
        print("Epoch: {} | Meta-Loss: {:.5f}".format(epoch, meta_loss.item()))
        wandb.log({"Meta-Loss": meta_loss}) # log the loss to wandb

        early_stopper(model.GNN.state_dict(), meta_loss)
        if early_stopper.early_stop:
            print("Stopping training...")
            print("Best Score: ", early_stopper.best_score)
            break

        optimizer.zero_grad()
        meta_loss.backward()
        optimizer.step()

