import argparse

def get_pretrain_args():
    parser = argparse.ArgumentParser(description='Args for Pretrain')

    # Datasets & Augmentation
    parser.add_argument("--task", type=str, default='node', help="if node level tasks")
    parser.add_argument("--dataset", type=str, nargs='+', default=["Amazon_Photo", "Amazon_Computer", "Amazon_Fraud"],
                        help="Datasets used for pretrain")
    parser.add_argument("--subgraphs", type=int, default=256, help="subgraph num for each dataset")
    parser.add_argument("--temperature", type=float, default=0.1, help="temperature for similarity calculation")
    parser.add_argument("--augment", type=str, nargs='+', default=["Subgraph", "Drop"], help="Augmentation for pretraining(Only support two methods)")
    parser.add_argument("--aug_ratio", type=float, nargs='+', default=[0.7, 0.3], help="Augmentation ratio")

    # Pretrained model
    parser.add_argument("--gnn_layer", type=int, default=2, help="layer num for gnn")
    parser.add_argument("--mlp_layer", type=int, default=2, help="layer num for mlp")
    parser.add_argument("--input_dim", type=int, default=100, help="input dimension")
    parser.add_argument("--hidden_dim", type=int, default=100, help="hidden dimension")
    parser.add_argument("--output_dim", type=int, default=100, help="output dimension(also dimension of projector and answering fuction)")
    parser.add_argument("--path", type=str, default="pretrained_gnn/Amazon_Photo+Amazon_Computer+Amazon_Fraud_Subgraph+Drop_GCN_2.pth", help="model saving path")

    # Bridge nodes
    parser.add_argument("--node_num", type=int, default=16, help="num of bridge nodes")
    parser.add_argument("--node_dim", type=int, default=100, help="feature dimension of bridge nodes, should be same as input dimension of subgraph nodes")
    parser.add_argument("--threshold", type=float, default=0.1, help="threshold for connecting nodes")

    # Pretrain Process
    parser.add_argument("--batch_size", type=int, default=8, help="subgraph num of one batch")
    parser.add_argument("--lr", type=float, default=0.001, help="learning rate for pretraining")
    parser.add_argument("--decay", type=float, default=0.0001, help="weight decay for pretraining")
    parser.add_argument("--max_epoches", type=int, default=300, help="max epoches for pretraining")
    parser.add_argument("--loss_bias", type=float, default=10.0, help="extra loss that will be added to contrastive loss calculation")

    # Trainging enviorment
    parser.add_argument("--gpu", type=int, default=-1, help="GPU id to use, -1 for CPU")
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument("--patience", type=int, default=10, help="early stop steps")
    
    args = parser.parse_args()
    return args


def get_downstream_args():
    parser = argparse.ArgumentParser(description='Args for Pretrain')

    # Datasets & Models
    parser.add_argument("--task", type=str, default='node', help="if node level tasks")
    parser.add_argument("--dataset", type=str, default="Amazon_Computer",help="Datasets used for downstream tasks")
    parser.add_argument("--subgraph_node_num", type=int, default="200",help="node num for each subgraph generated from dataset")
    parser.add_argument("--pretrained_model", type=str, default="pretrained_gnn/Amazon_Photo+Amazon_Computer+Amazon_Fraud_Subgraph+Drop_GCN_2.pth", help="pretrained model path")
    parser.add_argument("--shot", type=int, default=100, help="shot for few-shot learning")
    parser.add_argument("--k_hop", type=int, default=2, help="k-hop subgraph")
    parser.add_argument("--input_dim", type=int, default=100, help="input dimension")
    parser.add_argument("--gnn_layer", type=int, default=2, help="layer num for gnn")
    parser.add_argument("--mlp_layer", type=int, default=2, help="layer num for mlp")
    parser.add_argument("--hidden_dim", type=int, default=100, help="hidden dimension")
    parser.add_argument("--output_dim", type=int, default=100, help="output dimension(also dimension of projector and answering fuction)")

    # Prompt
    parser.add_argument("--new_pool", type=int, default=1, help="whether needs a new prompt pool, 1 is for 'Yes', and 0 is for 'No'")
    parser.add_argument("--if_train", type=int, default=1, help="0 is for freeze prompt pool, 1 is for activate the last component.")
    parser.add_argument("--prompt_num", type=int, default=10, help="prompt num for each component")
    parser.add_argument("--prompt_dim", type=int, default=100, help="dimension of prompt, should be same as hidden_dim")
    parser.add_argument("--prompt_layers", type=int, default=-1, help="-1 is for shallow prompt, other >0 values are for deep prompt(should be same as gnn_layer)")
    parser.add_argument("--prompt_path", type=str, default='downstream_model/prompt_pool_0.pth', help="prompt pool saving path")
    parser.add_argument("--answering_path", type=str, default='downstream_model/answering_0.pth', help="answering function saving path")

    # Downstream Tasks
    parser.add_argument("--lr", type=float, default=0.01, help="learning rate for downstream training")
    parser.add_argument("--decay", type=float, default=1e-4, help="weight decay for downstream training")
    parser.add_argument("--max_epoches", type=int, default=1000, help="max epoches for downstream training")

    # Trainging enviorment
    parser.add_argument("--gpu", type=int, default=-1, help="GPU id to use, -1 for CPU")
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument("--patience", type=int, default=10, help="early stop steps")
    
    args = parser.parse_args()
    return args