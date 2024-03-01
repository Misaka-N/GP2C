import argparse

def get_pretrain_args():
    parser = argparse.ArgumentParser(description='Args for Pretrain')
    parser.add_argument("--task", type=str, default='node', help="if node level tasks")
    parser.add_argument("--dataset", type=str, nargs='+', default=["Amazon_Photo", "Amazon_Computer", "Amazon_Fraud"],
                        help="Datasets used for pretrain")
    
    # Pretrain model
    parser.add_argument("--gnn_layer", type=int, default=2, help="layer num for gnn")
    parser.add_argument("--mlp_layer", type=int, default=2, help="layer num for mlp")
    parser.add_argument("--input_dim", type=int, default=128, help="input dimension")
    parser.add_argument("--hidden_dim", type=int, default=256, help="hidden dimension")
    parser.add_argument("--output_dim", type=int, default=64, help="output dimension")
    parser.add_argument("--subgraphs", type=int, default=100, help="subgraph num for each dataset")

    parser.add_argument("--gpu", type=int, default=-1, help="GPU id to use, -1 for CPU")
    parser.add_argument("--seed", type=int, default=3407, help="random seed")
    
    args = parser.parse_args()
    return args