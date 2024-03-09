import wandb
import torch
import numpy as np
import torch.nn as nn
import random
from tqdm import tqdm
from models.GIN import GIN
from models.GCN import GCN
import torch.optim as optim
import torch.nn.functional as F
from models.prompt import PromptComponent
from collections import defaultdict
from utils.args import get_downstream_args
from utils.tools import set_random
from utils.dataloader import pretrain_dataloader
from torch_geometric.data import Data
from torch_geometric.utils import k_hop_subgraph, degree, subgraph
from torch_geometric.loader import DataLoader
from utils.tools import EarlyStopping
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, recall_score, average_precision_score


def get_induced_graph(data, num_classes, shot, k):
    node_idx = range(data.x.size(0))
    edge_index = data.edge_index
    class_to_subgraphs = {i: [] for i in range(num_classes)}
    
    for node, label in tqdm(zip(node_idx, data.y), total=data.x.size(0)):
        subgraph_node_idx, subgraph_edge_index, _, _ = k_hop_subgraph(node, 1, edge_index, relabel_nodes=True)
        class_to_subgraphs[label.item()].append(Data(x=data.x[subgraph_node_idx], edge_index=subgraph_edge_index, y=label))
    
    train_list, test_list = [], []

    for label in range(num_classes):
        subgraphs = class_to_subgraphs[label]
        if len(subgraphs) < shot:
            raise ValueError("Fail to get {} shot. The subgraph num is {}".format({shot, subgraphs.shape[0]}))

        train_list.extend(subgraphs[:shot])
        test_list.extend(subgraphs[shot:])

    random.shuffle(train_list)
    random.shuffle(test_list)
    
    return DataLoader(dataset=train_list, batch_size=1, shuffle=False, num_workers=0), DataLoader(dataset=test_list, batch_size=1, shuffle=False, num_workers=0)


def gram_loss(prompt):
    gram_matrix = torch.matmul(prompt, prompt.t())
    diagonals = gram_matrix.diag()
    diagonal_loss = torch.sum((diagonals - 1)**2)
    off_diagonals = gram_matrix.fill_diagonal_(0)
    off_diagonal_loss = torch.sum(torch.abs(off_diagonals))
    loss = diagonal_loss + off_diagonal_loss

    return loss


def loss_fn(cross_fn, predict, label, prompt):
    cross_loss = cross_fn(predict, label)
    cross_loss += args.ortho_weight * ortho_penalty(prompt)
    return cross_loss


def ortho_penalty(t):
    return ((t @t.T - torch.eye(t.shape[0]).cuda())**2).mean()


if __name__ == "__main__":
    args = get_downstream_args()

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

    # Get downstream datasets
    # TODO: Deal with Graph Tasks(Now only support node tasks)
    if args.task == 'node': # Node level tasks
        print("---Downloading dataset: " + args.dataset + "---")
        data, dataname, num_classes = pretrain_dataloader(input_dim=args.input_dim, dataset=args.dataset)
        print("---Getting induced graphs: " + args.dataset + "---")
        train_set, val_set = get_induced_graph(data, num_classes, args.shot, args.k_hop)
    else:
        pass

    # Get prompt pool
    print("---Loading prompt pool: new pool: {}, train: {}---".format(args.new_pool, args.if_train))
    if args.new_pool:
        prompt_pool = PromptComponent(prompt_num=args.prompt_num, prompt_dim=args.prompt_dim, input_dim=args.input_dim, layers=args.prompt_layers)
    else:
        prompt_pool = torch.load(args.prompt_path)
    if args.if_train:
        prompt_pool.new_task_init()
        answering = nn.Sequential(
            nn.Linear(args.output_dim, args.output_dim),
            nn.PReLU(),
            nn.Linear(args.output_dim, num_classes))
        for _, layer in enumerate(answering):
            if isinstance(layer, nn.Linear):
                nn.init.normal_(layer.weight, mean=0.0, std=1.0)
                nn.init.normal_(layer.bias, mean=0.0, std=1.0)
    elif not args.if_train and args.new_pool:
        raise ValueError("Can't create a new pool but not train.")
    else:
        answering = torch.load(args.answering_path)


    if args.if_train:
        # Downstream tasks
        print("---Dealing with downstream task---")
        gnn = GCN(gcn_layer_num=args.gnn_layer, input_dim=args.input_dim, hid_dim=args.hidden_dim, out_dim=args.output_dim)

        # start a new wandb run to track this script
        wandb.init(
            # set the wandb project where this run will be logged
            project="l2s_" + args.dataset,
            # track hyperparameters and run metadata
            config=args.__dict__
            )
        # optimizer_ans = optim.Adam(answering.parameters(), lr=args.lr, weight_decay=args.decay)
        # optimizer_pro = optim.Adam(prompt_pool.parameters(), lr=args.lr, weight_decay=args.decay)
        optimizer = optim.Adam(list(prompt_pool.parameters()) + list(answering.parameters()), lr=args.lr, weight_decay=args.decay)
        early_stopper = EarlyStopping(path1=args.prompt_path, patience=args.patience, min_delta=0, path2=args.answering_path)
        cross_loss = nn.CrossEntropyLoss()

        for epoch in range(args.max_epoches):
            gnn.eval()

            # tune answering function
            # prompt_pool.eval()
            # answering.train()
            # predict, label, pred = [], [], []
            # for _, subgraph in enumerate(train_set):
            #     read_out = subgraph.x.mean(dim=0)
            #     summed_prompt = prompt_pool(read_out, args.if_train)
            #     predict_feat = gnn(subgraph.x, subgraph.edge_index, subgraph.batch, summed_prompt, args.prompt_layers).mean(dim=0)
            #     pre = answering(predict_feat.unsqueeze(0))

            #     pred.append(pre)
            #     predict.append(pre.argmax(dim=1))
            #     label.append(subgraph.y)

            # train_loss = loss_fn(cross_loss, torch.stack(pred).squeeze(1), torch.stack(label).squeeze(1))

            # optimizer_ans.zero_grad()
            # train_loss.backward()
            # optimizer_ans.step()
                
            # accuracy = accuracy_score(label, predict)
            # print("Epoch: {} | Answering Function | Loss: {:.4f} | ACC: {:.4f}".format(epoch, train_loss, accuracy))

            # tune prompt
            prompt_pool.train()
            answering.train()
            predict, label, pred = [], [], []
            for _, subgraph in enumerate(train_set):
                read_out = subgraph.x.mean(dim=0)
                summed_prompt = prompt_pool(read_out, args.if_train)
                predict_feat = gnn(subgraph.x, subgraph.edge_index, subgraph.batch, summed_prompt, args.prompt_layers).mean(dim=0)
                pre = answering(predict_feat.unsqueeze(0))

                pred.append(pre)
                predict.append(pre.argmax(dim=1))
                label.append(subgraph.y)

            train_loss = loss_fn(cross_loss, torch.stack(pred).squeeze(1), torch.stack(label).squeeze(1), prompt_pool.prompt[prompt_pool.task_cnt-1])

            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()
                
            accuracy = accuracy_score(label, predict)
            wandb.log({"train_loss": train_loss, "train_accuracy": accuracy})
            print("Epoch: {} | Prompt | Loss: {:.4f} | ACC: {:.4f}".format(epoch, train_loss, accuracy))

            early_stopper((prompt_pool, answering), train_loss.item())
            if early_stopper.early_stop:
                print("Stopping training...")
                print("Best Score: ", early_stopper.best_score)
                break

            if (epoch + 1) % 10 == 0: 
                # Evaluation
                gnn.eval()
                prompt_pool.eval()
                answering.eval()
                predict, pred, label = [], [], []
                for _, subgraph in tqdm(enumerate(val_set), desc="Evaluating Process"):
                    read_out = subgraph.x.mean(dim=0)
                    summed_prompt = prompt_pool(read_out, args.if_train)
                    predict_feat = gnn(subgraph.x, subgraph.edge_index, subgraph.batch, summed_prompt, args.prompt_layers).mean(dim=0)
                    pre = answering(predict_feat.unsqueeze(0))
                    predict.append(pre.argmax(dim=1))
                    pre = F.softmax(pre, dim=-1)
                    pred.append(pre.detach().squeeze())
                    label.append(subgraph.y)
                accuracy = accuracy_score(label, predict)
                auc = roc_auc_score(label, pred, multi_class='ovr')
                recall = recall_score(label, predict, average='macro')
                f1 = f1_score(label, predict, average='macro')
                ap = average_precision_score(label, pred)
                wandb.log({"val_accuracy": accuracy, "val_auc": auc, "val_recall": recall, "val_f1": f1})
                print("Epoch: {} | ACC: {:.4f} | AUC: {:.4f} | F1: {:.4f} | Recall : {:.4f} | AP: {:.4f}".format(epoch+1, accuracy, auc, recall, f1, ap))
        wandb.finish()

    # test on the best model
    print("Evaluating on the best model...")
    gnn.eval()
    prompt_pool = torch.load(args.prompt_path)
    answering = torch.load(args.answering_path)
    prompt_pool.eval()
    answering.eval()
    predict, pred, label = [], [], []
    for _, subgraph in tqdm(enumerate(val_set), desc="Evaluating Process"):
        read_out = subgraph.x.mean(dim=0)
        summed_prompt = prompt_pool(read_out, args.if_train)
        predict_feat = gnn(subgraph.x, subgraph.edge_index, subgraph.batch, summed_prompt, args.prompt_layers).mean(dim=0)
        pre = answering(predict_feat.unsqueeze(0))
        predict.append(pre.argmax(dim=1))
        pre = F.softmax(pre, dim=-1)
        pred.append(pre.detach().squeeze())
        label.append(subgraph.y)

    accuracy = accuracy_score(label, predict)
    auc = roc_auc_score(label, pred, multi_class='ovr')
    recall = recall_score(label, predict, average='macro')
    f1 = f1_score(label, predict, average='macro')
    ap = average_precision_score(label, pred)
    print("Epoch: {} | ACC: {:.4f} | AUC: {:.4f} | F1: {:.4f} | Recall : {:.4f} | AP: {:.4f}".format(epoch+1, accuracy, auc, recall, f1, ap))

    