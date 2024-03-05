import wandb
import torch
import torch.nn as nn
import random
from tqdm import tqdm
from models.GIN import GIN
import torch.optim as optim
from models.prompt import PromptComponent
from collections import defaultdict
from utils.args import get_downstream_args
from utils.tools import set_random
from utils.dataloader import pretrain_dataloader
from torch_geometric.data import Data
from torch_geometric.utils import k_hop_subgraph
from torch_geometric.loader import DataLoader
from utils.tools import EarlyStopping
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, recall_score, average_precision_score

def get_induced_graph(data, num_classes, shot, k):
    node_idx = range(data.x.size(0))
    edge_index = data.edge_index
    class_to_subgraphs = {i: [] for i in range(num_classes)}
    
    for node, label in tqdm(zip(node_idx, data.y), total=data.x.size(0)):
        subgraph_node_idx, subgraph_edge_index, _, _ = k_hop_subgraph(node, k, edge_index, relabel_nodes=True)
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


def loss_fn(cross_fn, predict, label):
    cross_loss = cross_fn(predict, label)

    return cross_loss


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
        answering = torch.nn.Sequential(
            torch.nn.Linear(args.output_dim, num_classes),
            torch.nn.Softmax(dim=1))
    elif not args.if_train and args.new_pool:
        raise ValueError("Can't create a new pool but not train.")
    else:
        answering = torch.load(args.answering_path)


    if args.if_train:
        # Downstream tasks
        print("---Dealing with downstream task---")
        gnn = torch.load(args.pretrained_model)
        for param in gnn.parameters(): # freeze GNN
            param.requires_grad = False

        # start a new wandb run to track this script
        wandb.init(
            # set the wandb project where this run will be logged
            project="l2s_" + args.dataset,
            # track hyperparameters and run metadata
            config=args.__dict__
            )

        optimizer = optim.Adam(list(answering.parameters()) + list(prompt_pool.parameters()), lr=args.lr, weight_decay=args.decay)
        early_stopper = EarlyStopping(path1=args.prompt_path, patience=args.patience, min_delta=0, path2=args.answering_path)
        cross_loss = nn.CrossEntropyLoss()

        for epoch in range(args.max_epoches):
            gnn.train()
            prompt_pool.train()
            answering.train()
            predict, label = [], []
            running_loss = 0
            for _, subgraph in enumerate(train_set):
                read_out = subgraph.x.mean(dim=0)
                summed_prompt = prompt_pool(read_out, args.if_train)
                print(summed_prompt[0].grad)
                quit()
                predict_feat = gnn(subgraph.x, subgraph.edge_index, subgraph.batch, summed_prompt, args.prompt_layers).mean(dim=0)
                pre = answering(predict_feat.unsqueeze(0))

                train_loss = loss_fn(cross_loss, pre, subgraph.y)
                running_loss += train_loss.item()

                optimizer.zero_grad()
                train_loss.backward()
                optimizer.step()
                
                predict.append(pre.argmax(dim=1))
                label.append(subgraph.y)

            running_loss /= len(train_set)
            accuracy = accuracy_score(label, predict)
            wandb.log({"train_loss": running_loss, "train_accuracy": accuracy})
            print("Epoch: {} | Loss: {:.4f}".format(epoch, running_loss))

            early_stopper((prompt_pool,answering), running_loss)
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
                for _, subgraph in enumerate(val_set):
                    read_out = subgraph.x.mean(dim=0)
                    summed_prompt = prompt_pool(read_out, args.if_train)
                    predict_feat = gnn(subgraph.x, subgraph.edge_index, subgraph.batch, summed_prompt, args.prompt_layers).mean(dim=0)
                    pre = answering(predict_feat.unsqueeze(0))
                    pred.append(pre.detach().squeeze())
                    predict.append(pre.argmax(dim=1))
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
    print("Testing on the best model...")
    gnn.eval()
    prompt_pool = torch.load(args.prompt_path)
    answering = torch.load(args.answering_path)
    prompt_pool.eval()
    answering.eval()
    predict, pred, label = [], [], []
    for _, subgraph in enumerate(val_set):
        read_out = subgraph.x.mean(dim=0)
        summed_prompt = prompt_pool(read_out, args.if_train)
        predict_feat = gnn(subgraph.x, subgraph.edge_index, subgraph.batch, summed_prompt, args.prompt_layers).mean(dim=0)
        pre = answering(predict_feat.unsqueeze(0))
        pred.append(pre.detach().squeeze())
        predict.append(pre.argmax(dim=1))
        label.append(subgraph.y)

    accuracy = accuracy_score(label, predict)
    auc = roc_auc_score(label, pred, multi_class='ovr')
    recall = recall_score(label, predict, average='macro')
    f1 = f1_score(label, predict, average='macro')
    ap = average_precision_score(label, pred)
    print("Epoch: {} | ACC: {:.4f} | AUC: {:.4f} | F1: {:.4f} | Recall : {:.4f} | AP: {:.4f}".format(epoch+1, accuracy, auc, recall, f1, ap))()

    