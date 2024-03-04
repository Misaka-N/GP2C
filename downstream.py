import torch
from models.GIN import GIN
from utils.args import get_downstream_args
from utils.tools import set_random
from utils.dataloader import pretrain_dataloader
from torch_geometric.data import Data
from torch_geometric.utils import k_hop_subgraph

def load_model(pretrained_model_name:str):
    pretrained_GNN = torch.load('pretrained_gnn/' + pretrained_model_name + '.pth')
    # pretrained_GNN.load_state_dict(torch.load('/pretrained_model/' + pretrained_model_name + '.pth'))

    return pretrained_GNN

def get_induced_graph(data:Data, num_classes:int, shot:int):
    k = args.k_hop

    # Generate k-hop subgraph for each node
    node_idx = range(data.x.size(0))
    edge_index = data.edge_index
    subgraph_list = []
    for node in node_idx:
        subgraph = k_hop_subgraph(node, k, edge_index)
        subgraph_list.append(subgraph)

    for subgraph in subgraph_list:
        subgraph_data = data[subgraph]
        subgraph_representation = subgraph_data.x


    return



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


    gnn = load_model(args.pretrained_model)

    # Get pretrain datasets
    # TODO: Deal with Graph Tasks(Now only support node tasks)
    if args.task == 'node': # Node level tasks
        for dataset in args.dataset:
            print("---Downloading dataset: " + dataset + "---")
            data, dataname, num_classes = pretrain_dataloader(input_dim=args.input_dim, dataset=dataset)
            get_induced_graph(data, num_classes, args.shot)
    else:
        pass

