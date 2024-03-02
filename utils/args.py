import argparse

def get_pretrain_args():
    parser = argparse.ArgumentParser(description='Args for Pretrain')

    # Datasets & Augmentation
    parser.add_argument("--task", type=str, default='node', help="if node level tasks")
    parser.add_argument("--dataset", type=str, nargs='+', default=["Amazon_Photo", "Amazon_Computer", "Amazon_Fraud"],
                        help="Datasets used for pretrain")
    parser.add_argument("--subgraphs", type=int, default=256, help="subgraph num for each dataset")
    parser.add_argument("--temperature", type=float, default=0.1, help="temperature for similarity calculation")
    parser.add_argument("--augment", type=str, nargs='+', default=["Subgraph", "Drop"], help="Augmentation for pretraining")
    parser.add_argument("--aug_radio", type=float, nargs='+', default=[0.2, 0.2], help="Augmentation radio")

    # Pretrained model
    parser.add_argument("--gnn_layer", type=int, default=2, help="layer num for gnn")
    parser.add_argument("--mlp_layer", type=int, default=2, help="layer num for mlp")
    parser.add_argument("--input_dim", type=int, default=128, help="input dimension")
    parser.add_argument("--hidden_dim", type=int, default=256, help="hidden dimension")
    parser.add_argument("--output_dim", type=int, default=32, help="output dimension(also dimension of projector)")

    # Bridge nodes
    parser.add_argument("--node_num", type=int, default=16, help="num of bridge nodes")
    parser.add_argument("--node_group", type=int, default=256, help="group num of bridge nodes, should be same as the subgraph num of each dataset")
    parser.add_argument("--node_dim", type=int, default=128, help="feature dimension of bridge nodes, should be same as input dimension of subgraph nodes")
    parser.add_argument("--node_threshold", type=int, default=16, help="threshold for connecting nodes")

    # Pretrain Process
    parser.add_argument("--batch_size", type=int, default=8, help="subgraph num of one batch")
    parser.add_argument("--lr", type=float, default=0.01, help="learning rate for pretraining")
    parser.add_argument("--decay", type=float, default=0.0001, help="weight decay for pretraining")
    parser.add_argument("--max_epoches", type=int, default=200, help="max epoches for pretraining")

    # Trainging enviorment
    parser.add_argument("--gpu", type=int, default=-1, help="GPU id to use, -1 for CPU")
    parser.add_argument("--seed", type=int, default=3407, help="random seed")
    parser.add_argument("--patience", type=int, default=10, help="early stop steps")
    parser.add_argument("--id", type=int, default=0, help="just for identifying models")
    
    args = parser.parse_args()
    return args