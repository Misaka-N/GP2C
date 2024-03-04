import wandb
import torch
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


def get_induced_graph(data, num_classes, shot, k):
    node_idx = range(data.x.size(0))
    edge_index = data.edge_index
    class_to_subgraphs = {i: [] for i in range(num_classes)}
    
    for node, label in tqdm(zip(node_idx, data.y), total=data.x.size(0)):
        subgraph_node_idx, subgraph_edge_index, _, _ = k_hop_subgraph(node, k, edge_index, relabel_nodes=False)
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
        train_set, val_set = get_induced_graph(data,num_classes, args.shot, args.k_hop)
    else:
        pass

    # Get prompt pool
    print("---Loading prompt pool: new pool: {}, train: {}---".format(args.new_pool, args.if_train))
    if args.new_pool:
        prompt_pool = PromptComponent(prompt_num=args.prompt_num, prompt_dim=args.prompt_dim, layers=args.prompty_layers)
    else:
        # read prompts
        pass
    if args.if_train:
        prompt_pool.new_task_init()
    elif not args.if_train and args.new_pool:
        raise ValueError("Can't create a new pool but not train.")

    # Downstream tasks
    print("---Dealing with downstream task---")
    gnn = torch.load(args.pretrained_model)
    for param in gnn.parameters(): # freeze GNN
        param.requires_grad = False
    optimizer = optim.Adam(gnn.parameters(), lr=args.lr, weight_decay=args.decay)
    early_stopper = EarlyStopping(path=args.path, patience=args.patience, min_delta=0)

    # start a new wandb run to track this script
    wandb.init(
        # set the wandb project where this run will be logged
        project="l2s_" + args.dataset,
        # track hyperparameters and run metadata
        config=args.__dict__
        )

    gnn.train()
    prompt_pool.train()
    for epoch in range(args.max_epoched):
        predict, label = [], []
        for subgraph in enumerate(train_set):
            read_out = subgraph.x.mean()
            summed_prompt = prompt_pool(read_out, args.if_train)
            predict_feat = gnn(subgraph.x, subgraph.edge_index, subgraph.batch, summed_prompt).mean()
        