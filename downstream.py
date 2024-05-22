import wandb
import torch
import numpy as np
import torch.nn as nn
import networkx as nx
import random
from tqdm import tqdm
from models.GIN import GIN
from models.GCN import GCN
import torch.optim as optim
import torch.nn.functional as F
from models.prompt import PromptComponent
from collections import defaultdict
from utils.args import get_downstream_args, get_pretrain_args
from utils.tools import set_random
from utils.dataloader import pretrain_dataloader
from torch_geometric.nn import global_mean_pool
from torch_geometric.data import Data
from torch_geometric.utils import k_hop_subgraph, degree, subgraph, from_networkx, to_networkx, to_dense_batch, to_undirected
from torch_geometric.loader import DataLoader
from utils.tools import EarlyStopping, label_smoothing
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, recall_score, average_precision_score
from cdlib import algorithms, viz
import matplotlib.pyplot as plt

import os
os.environ["WANDB_MODE"] = "offline"

def get_induced_graph(data, num_classes, shot, k, max_node_num, batch_size):
    edge_index = data.edge_index
    class_to_subgraphs = {i: [] for i in range(num_classes)}
    class_to_center_node = {i: [] for i in range(num_classes)}

    # the following code is used to select the most important nodes in the subgraph
    if max_node_num != -1:
        pr = nx.pagerank(to_networkx(data), alpha=0.85)

    for node, label in tqdm(enumerate(data.y), total=data.x.size(0)):
        subgraph_node_idx, subgraph_edge_idx, mapping, _ = k_hop_subgraph(node, k, edge_index, relabel_nodes=True)
        center_node_new_idx = mapping.item()

        # the following code is used to select the most important nodes in the subgraph
        if max_node_num !=-1 and len(subgraph_node_idx) > max_node_num:
            important_nodes = sorted((node for node in subgraph_node_idx if node != node), key=lambda x: pr[x], reverse=True)[:max_node_num - 1]
            important_nodes.append(node)
            subgraph_node_idx = torch.tensor(important_nodes, dtype=torch.long)
            subgraph_edge_idx, _ = subgraph(subgraph_node_idx, edge_index, relabel_nodes=True)
            center_node_new_idx = (subgraph_node_idx == node).nonzero(as_tuple=True)[0].item()

        class_to_subgraphs[label.item()].append(Data(x=data.x[subgraph_node_idx], edge_index=subgraph_edge_idx, y=label))
        class_to_center_node[label.item()].append(center_node_new_idx)

    train_list, test_list, train_idx, test_idx = [], [], [], []

    for label, subgraphs in class_to_subgraphs.items():
        center_nodes = class_to_center_node[label]
        num_subgraphs = len(subgraphs)
        if num_subgraphs < shot:
            raise ValueError(f"Fail to get {shot} shot. The subgraph num is {len(subgraphs)}")

        for i, item in enumerate(subgraphs):
            if i < shot:
                train_list.append(item)
                train_idx.append(center_nodes[i])
            else:
                test_list.append(item)
                test_idx.append(center_nodes[i])

    train_indices = list(range(len(train_list))) 
    test_indices = list(range(len(test_list))) 
    random.shuffle(train_indices)
    random.shuffle(test_indices)
    shuffled_train_list = [train_list[i] for i in train_indices]
    shuffled_train_idx = [train_idx[i] for i in train_indices]
    shuffled_test_list = [test_list[i] for i in test_indices]
    shuffled_test_idx = [test_idx[i] for i in test_indices]

    print("---Split val set into val and test---")
    shuffled_val_list = shuffled_test_list[:len(shuffled_test_list) // 2]
    shuffled_val_idx = shuffled_test_idx[:len(shuffled_test_idx) // 2]
    shuffled_test_list = shuffled_test_list[len(shuffled_test_list) // 2:]
    shuffled_test_idx = shuffled_test_idx[len(shuffled_test_idx) // 2:]
    
    
    return DataLoader(dataset=shuffled_train_list, batch_size=batch_size, shuffle=False, num_workers=0), DataLoader(dataset=shuffled_val_list, batch_size=1, shuffle=False, num_workers=0), DataLoader(dataset=shuffled_test_list, batch_size=1, shuffle=False, num_workers=0),\
           DataLoader(dataset=shuffled_train_idx, batch_size=batch_size, shuffle=False, num_workers=0), DataLoader(dataset=shuffled_val_idx, batch_size=1, shuffle=False, num_workers=0), DataLoader(dataset=shuffled_test_idx, batch_size=1, shuffle=False, num_workers=0)

def get_graphs(data, num_classes, shot, batch_size):
    train_set = []
    val_set = []
    test_set = []
    
    class_samples = {i: [] for i in range(num_classes)}
    for i in range(len(data)):
        sample = data[i]
        label = sample.y.item()
        class_samples[label].append(sample)
    
    for label, samples in class_samples.items():
        random.shuffle(samples) 
        train_set.extend(samples[:shot]) 
        
        remaining_samples = samples[shot:]
        split_idx = len(remaining_samples) // 2
        
        val_set.extend(remaining_samples[:split_idx])
        test_set.extend(remaining_samples[split_idx:])
    print("Train set size: ", len(train_set), "Val set size: ", len(val_set), "Test set size: ", len(test_set))
    return DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True, num_workers=0), DataLoader(dataset=val_set, batch_size=1, shuffle=True, num_workers=0), DataLoader(dataset=test_set, batch_size=1, shuffle=True, num_workers=0) 


def loss_fn(loss_fn, predict, label):
    loss = loss_fn(predict, label)
    return loss

if __name__ == "__main__":
    args = get_downstream_args()

    print("PyTorch version:", torch.__version__)

    if torch.cuda.is_available() and args.gpu != -1:
        print("CUDA is available")
        print("CUDA version:", torch.version.cuda)
        device = torch.device("cuda:"+str(args.gpu))
        set_random(args.seed, True)
    else:
        print("CUDA is not available")
        device = torch.device("cpu")
        set_random(args.seed, False)
    print("Device:", device)

    # Get downstream datasets
    print("---Downloading dataset: " + args.dataset + "---")
    data, dataname, num_classes = pretrain_dataloader(input_dim=args.input_dim, dataset=args.dataset)
    if args.task == 'node': # Node level tasks
        print("---Getting induced graphs: " + args.dataset + "---")
        train_set, val_set, test_set, train_idx, val_idx, test_idx = get_induced_graph(data, num_classes, args.shot, args.k_hop, args.k_hop_nodes, args.batch_size)
    elif args.task == 'graph':
        print("---Getting graphs: " + args.dataset + "---")
        train_set, val_set, test_set = get_graphs(data, num_classes, args.shot, args.batch_size)
    else:
        raise NotImplementedError("Unsupported Tasks.")



    if args.task == 'node':
        # Get prompt pool
        print("---Creating new prompt pool---")
        prompt_pool = PromptComponent(prompt_num=args.prompt_num, prompt_dim=args.prompt_dim, input_dim=args.input_dim).to(device)
        prompt_pool.new_task_init(device)

        # answering head
        answering = nn.Sequential(
            nn.Linear(2 * args.output_dim, args.output_dim),
            nn.PReLU(),
            nn.Linear(args.output_dim, num_classes)).to(device)
        for _, layer in enumerate(answering):
            if isinstance(layer, nn.Linear):
                nn.init.normal_(layer.weight, mean=0.0, std=1.0)
                nn.init.normal_(layer.bias, mean=0.0, std=1.0)

        # Downstream tasks
        print("---Dealing with downstream task---")
        gnn = GCN(gcn_layer_num=args.gnn_layer, input_dim=args.input_dim, hid_dim=args.hidden_dim, out_dim=args.output_dim).to(device)
        gnn.load_state_dict(torch.load(args.pretrained_model))

        optimizer = optim.Adam(list(prompt_pool.parameters()) + list(answering.parameters()), lr=args.lr, weight_decay=args.decay)
        early_stopper = EarlyStopping(path1=args.prompt_path, patience=args.patience, min_delta=0)
        cross_loss = nn.CrossEntropyLoss()
        kl_loss = torch.nn.KLDivLoss(reduction='batchmean')

        # start a new wandb
        wandb.init(
            project="l2s_downstream_" + dataname,
            config=args.__dict__,
            name=str(args.seed)
            )
    

        for epoch in range(args.max_epoches):
            gnn.eval()
            prompt_pool.train()
            answering.train()
            for step, (node_subgraph, node_idx) in enumerate(zip(train_set, train_idx)):
                pred, tot_loss = [], []

                predict_feat = gnn(node_subgraph.x.to(device), node_subgraph.edge_index.to(device), node_subgraph.batch.to(device))
                predict_feat, mask = to_dense_batch(predict_feat, node_subgraph.batch.to(device))

                for i in range(predict_feat.size(0)):
                    graph_feat = predict_feat[i][mask[i]]

                    # read_out = graph_feat[node_idx[i]]
                    read_out = graph_feat.mean(dim=0)
                    summed_prompt = prompt_pool(read_out)

                    sim_matrix = torch.matmul(graph_feat, summed_prompt.t())
                    node_emb = graph_feat * torch.matmul(sim_matrix, summed_prompt)

                    graph_emb = torch.concat((node_emb[node_idx[i]], torch.mean(node_emb, dim=0)), dim=-1)
                    pre = answering(graph_emb).cpu()

                    pred.append(pre)

                soft_label = label_smoothing(node_subgraph.y, args.label_smoothing, num_classes)
                pred = torch.stack(pred, dim=0)

                train_loss = loss_fn(kl_loss, F.log_softmax(pred, dim=-1), soft_label) / len(pred)
                tot_loss.append(train_loss.item())

                optimizer.zero_grad()
                train_loss.backward()
                optimizer.step()

                _, predict = torch.max(pred, dim=1)
                _, labels = torch.max(soft_label, dim=1)
                accuracy = accuracy_score(labels.cpu().numpy(), predict.cpu().numpy())
                wandb.log({"train_loss": train_loss.item(), "train_accuracy": accuracy})
                print("Epoch: {} | Step: {} | Loss: {:.4f} | ACC: {:.4f}".format(epoch, step, train_loss.item(), accuracy))

            if (epoch + 1) % 8 == 0: 
                # Evaluation
                gnn.eval()
                prompt_pool.eval()
                answering.eval()
                predict, pred, label = [], [], []
                for _, (node_subgraph, node_idx) in tqdm(enumerate(zip(val_set, val_idx)), desc="Evaluating Process"):
                    predict_feat = gnn(node_subgraph.x.to(device), node_subgraph.edge_index.to(device), node_subgraph.batch.to(device))

                    read_out = predict_feat.mean(dim=0)
                    # read_out = predict_feat[node_idx]
                    summed_prompt = prompt_pool(read_out)

                    sim_matrix = torch.matmul(predict_feat, summed_prompt.t())
                    node_emb = predict_feat * torch.matmul(sim_matrix, summed_prompt)

                    graph_emb = torch.concat((node_emb[node_idx], global_mean_pool(node_emb, node_subgraph.batch.long().to(device))), dim=-1)
                    
                    pre = answering(graph_emb).cpu()

                    predict.append(pre.argmax(dim=1))
                    pre = F.softmax(pre, dim=-1)
                    pred.append(pre.detach())
                    label.append(node_subgraph.y)

                pred = torch.stack(pred).squeeze()
                predict = torch.stack(predict).squeeze()
                label = torch.stack(label).squeeze() 

                # metrics
                accuracy = accuracy_score(label, predict)
                if num_classes == 2:
                    auc = roc_auc_score(label, pred[ :,1]) # for binary classification
                else:
                    auc = roc_auc_score(label, pred, multi_class='ovr')
                recall = recall_score(label, predict, average='macro')
                f1 = f1_score(label, predict, average='macro')
                if num_classes == 2:
                    ap = average_precision_score(label, pred[ :,1]) # for binary classification
                else:
                    ap = average_precision_score(label, pred)

                early_stopper((prompt_pool, answering), -(accuracy+auc+recall+f1+ap))
                if early_stopper.early_stop:
                    print("Stopping training...")
                    break

                wandb.log({"val_accuracy": accuracy, "val_auc": auc, "val_recall": recall, "val_f1": f1, 'val_ap': ap})
                print("Epoch: {} | ACC: {:.4f} | AUC: {:.4f} | F1: {:.4f} | Recall : {:.4f} | AP: {:.4f}".format(epoch+1, accuracy, auc, f1, recall, ap))

        # test on the best model
        print("Evaluating on the best model...")
        gnn.eval()
        prompt_pool,answering = torch.load(args.prompt_path)
        torch.save(prompt_pool, args.save_component_path) # save the component for L2S Plus
        prompt_pool.eval()
        answering.eval()
        predict, pred, label = [], [], []
        for _, (node_subgraph, node_idx) in tqdm(enumerate(zip(test_set, test_idx)), desc="Evaluating Process"):
            predict_feat = gnn(node_subgraph.x.to(device), node_subgraph.edge_index.to(device), node_subgraph.batch.to(device))

            read_out = predict_feat.mean(dim=0)
            # read_out = predict_feat[node_idx]
            summed_prompt = prompt_pool(read_out)

            sim_matrix = torch.matmul(predict_feat, summed_prompt.t())
            node_emb = predict_feat * torch.matmul(sim_matrix, summed_prompt)

            graph_emb = torch.concat((node_emb[node_idx], global_mean_pool(node_emb, node_subgraph.batch.long().to(device))), dim=-1)
            
            pre = answering(graph_emb).cpu()

            predict.append(pre.argmax(dim=1))
            pre = F.softmax(pre, dim=-1)
            pred.append(pre.detach())
            label.append(node_subgraph.y)

        pred = torch.stack(pred).squeeze()
        predict = torch.stack(predict).squeeze()
        label = torch.stack(label).squeeze()   

        # metrics
        accuracy = accuracy_score(label, predict)
        if num_classes == 2:
            auc = roc_auc_score(label, pred[ :,1]) # for binary classification
        else:
            auc = roc_auc_score(label, pred, multi_class='ovr')
        recall = recall_score(label, predict, average='macro')
        f1 = f1_score(label, predict, average='macro')
        if num_classes == 2:
            ap = average_precision_score(label, pred[ :,1]) # for binary classification
        else:
            ap = average_precision_score(label, pred)
        print("Final: | ACC: {:.4f} | AUC: {:.4f} | F1: {:.4f} | Recall : {:.4f} | AP: {:.4f}".format(accuracy, auc, f1, recall, ap))
        wandb.finish()
  
    elif args.task == 'graph':
       # Get prompt pool
        print("---Creating new prompt pool---")
        prompt_pool = PromptComponent(prompt_num=args.prompt_num, prompt_dim=args.prompt_dim, input_dim=args.input_dim).to(device)
        prompt_pool.new_task_init(device)

        # answering head
        answering = nn.Sequential(
            nn.Linear(args.output_dim, args.output_dim),
            nn.PReLU(),
            nn.Linear(args.output_dim, num_classes)).to(device)
        for _, layer in enumerate(answering):
            if isinstance(layer, nn.Linear):
                nn.init.normal_(layer.weight, mean=0.0, std=1.0)
                nn.init.normal_(layer.bias, mean=0.0, std=1.0)

        # Downstream tasks
        print("---Dealing with downstream task---")
        gnn = GCN(gcn_layer_num=args.gnn_layer, input_dim=args.input_dim, hid_dim=args.hidden_dim, out_dim=args.output_dim).to(device)
        gnn.load_state_dict(torch.load(args.pretrained_model))

        optimizer = optim.Adam(list(prompt_pool.parameters()) + list(answering.parameters()), lr=args.lr, weight_decay=args.decay)
        early_stopper = EarlyStopping(path1=args.prompt_path, patience=args.patience, min_delta=0)
        cross_loss = nn.CrossEntropyLoss()
        kl_loss = torch.nn.KLDivLoss(reduction='batchmean')

        # start a new wandb
        wandb.init(
            project="l2s_downstream_" + dataname,
            config=args.__dict__,
            name=str(args.seed)
            )

        for epoch in range(args.max_epoches):
            gnn.eval()
            prompt_pool.train()
            answering.train()
            for step, node_subgraph in enumerate(train_set):
                pred, tot_loss = [], []

                predict_feat = gnn(node_subgraph.x.to(device), node_subgraph.edge_index.to(device), None)
                predict_feat, mask = to_dense_batch(predict_feat, node_subgraph.batch.to(device))

                for i in range(predict_feat.size(0)):
                    graph_feat = predict_feat[i][mask[i]]

                    # read_out = graph_feat[node_idx[i]]
                    read_out = graph_feat.mean(dim=0)
                    summed_prompt = prompt_pool(read_out)

                    sim_matrix = torch.matmul(graph_feat, summed_prompt.t())
                    node_emb = graph_feat * torch.matmul(sim_matrix, summed_prompt)

                    graph_emb = torch.mean(node_emb, dim=0)
                    pre = answering(graph_emb).cpu()

                    pred.append(pre)

                soft_label = label_smoothing(node_subgraph.y, args.label_smoothing, num_classes)
                pred = torch.stack(pred, dim=0)

                train_loss = loss_fn(kl_loss, F.log_softmax(pred, dim=-1), soft_label) / len(pred)
                tot_loss.append(train_loss.item())

                optimizer.zero_grad()
                train_loss.backward()
                optimizer.step()

                _, predict = torch.max(pred, dim=1)
                _, labels = torch.max(soft_label, dim=1)
                accuracy = accuracy_score(labels.cpu().numpy(), predict.cpu().numpy())
                wandb.log({"train_loss": train_loss.item(), "train_accuracy": accuracy})
                print("Epoch: {} | Step: {} | Loss: {:.4f} | ACC: {:.4f}".format(epoch, step, train_loss.item(), accuracy))

            if (epoch + 1) % 2 == 0: 
                # Evaluation
                gnn.eval()
                prompt_pool.eval()
                answering.eval()
                predict, pred, label = [], [], []
                for _, node_subgraph in tqdm(enumerate(val_set), desc="Evaluating Process"):
                    predict_feat = gnn(node_subgraph.x.to(device), node_subgraph.edge_index.to(device),None)

                    read_out = predict_feat.mean(dim=0)
                    # read_out = predict_feat[node_idx]
                    summed_prompt = prompt_pool(read_out)

                    sim_matrix = torch.matmul(predict_feat, summed_prompt.t())
                    node_emb = predict_feat * torch.matmul(sim_matrix, summed_prompt)

                    graph_emb =  global_mean_pool(node_emb, node_subgraph.batch.long().to(device))
                    
                    pre = answering(graph_emb).cpu()

                    predict.append(pre.argmax(dim=1))
                    pre = F.softmax(pre, dim=-1)
                    pred.append(pre.detach())
                    label.append(node_subgraph.y)

                pred = torch.stack(pred).squeeze()
                predict = torch.stack(predict).squeeze()
                label = torch.stack(label).squeeze() 

                # metrics
                accuracy = accuracy_score(label, predict)
                if num_classes == 2:
                    auc = roc_auc_score(label, pred[ :,1]) # for binary classification
                else:
                    auc = roc_auc_score(label, pred, multi_class='ovr')
                recall = recall_score(label, predict, average='macro')
                f1 = f1_score(label, predict, average='macro')
                if num_classes == 2:
                    ap = average_precision_score(label, pred[ :,1]) # for binary classification
                else:
                    ap = average_precision_score(label, pred)

                early_stopper((prompt_pool, answering), -(accuracy+auc+recall+f1+ap))
                if early_stopper.early_stop:
                    print("Stopping training...")
                    break

                wandb.log({"val_accuracy": accuracy, "val_auc": auc, "val_recall": recall, "val_f1": f1, 'val_ap': ap})
                print("Epoch: {} | ACC: {:.4f} | AUC: {:.4f} | F1: {:.4f} | Recall : {:.4f} | AP: {:.4f}".format(epoch+1, accuracy, auc, f1, recall, ap))

        # test on the best model
        print("Evaluating on the best model...")
        gnn.eval()
        prompt_pool,answering = torch.load(args.prompt_path)
        torch.save(prompt_pool, args.save_component_path) # save the component for L2S Plus
        prompt_pool.eval()
        answering.eval()
        predict, pred, label = [], [], []
        for _, node_subgraph in tqdm(enumerate(test_set), desc="Evaluating Process"):
            predict_feat = gnn(node_subgraph.x.to(device), node_subgraph.edge_index.to(device), node_subgraph.batch.to(device))

            read_out = predict_feat.mean(dim=0)
            # read_out = predict_feat[node_idx]
            summed_prompt = prompt_pool(read_out)

            sim_matrix = torch.matmul(predict_feat, summed_prompt.t())
            node_emb = predict_feat * torch.matmul(sim_matrix, summed_prompt)

            graph_emb =global_mean_pool(node_emb, node_subgraph.batch.long().to(device))
            
            pre = answering(graph_emb).cpu()

            predict.append(pre.argmax(dim=1))
            pre = F.softmax(pre, dim=-1)
            pred.append(pre.detach())
            label.append(node_subgraph.y)

        pred = torch.stack(pred).squeeze()
        predict = torch.stack(predict).squeeze()
        label = torch.stack(label).squeeze()   

        # metrics
        accuracy = accuracy_score(label, predict)
        if num_classes == 2:
            auc = roc_auc_score(label, pred[ :,1]) # for binary classification
        else:
            auc = roc_auc_score(label, pred, multi_class='ovr')
        recall = recall_score(label, predict, average='macro')
        f1 = f1_score(label, predict, average='macro')
        if num_classes == 2:
            ap = average_precision_score(label, pred[ :,1]) # for binary classification
        else:
            ap = average_precision_score(label, pred)
        print("Final: | ACC: {:.4f} | AUC: {:.4f} | F1: {:.4f} | Recall : {:.4f} | AP: {:.4f}".format(accuracy, auc, f1, recall, ap))
        wandb.finish()