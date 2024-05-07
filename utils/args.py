import argparse

def get_pretrain_args():
    parser = argparse.ArgumentParser(description='Args for Pretrain')

    # Datasets & Augmentation
    parser.add_argument("--task", type=str, default='node', help="if node level tasks")
    parser.add_argument("--dataset", type=str, nargs='+', default=["Coauthor_CS", "Coauthor_Physics"],
                        help="Datasets used for pretrain")
    parser.add_argument("--subgraphs", type=int, default=128, help="subgraph num for each dataset")
    parser.add_argument("--temperature", type=float, default=0.1, help="temperature for similarity calculation")
    parser.add_argument("--augment", type=str, nargs='+', default=['Subgraph', 'Anonymize', 'Drop', 'Perturb', 'Mask'], help="Augmentation for pretraining")
    parser.add_argument("--aug_ratio", type=float, nargs='+', default=[0.45, 0.45, 0.45, 0.45, 0.45], help="Augmentation ratio")

    # Pretrained model
    parser.add_argument("--gnn_layer", type=int, default=2, help="layer num for gnn")
    parser.add_argument("--mlp_layer", type=int, default=2, help="layer num for mlp")
    parser.add_argument("--input_dim", type=int, default=500, help="input dimension")
    parser.add_argument("--hidden_dim", type=int, default=200, help="hidden dimension")
    parser.add_argument("--output_dim", type=int, default=200, help="output dimension(also dimension of projector and answering fuction)")
    parser.add_argument("--path", type=str, default="pretrained_gnn/Citation0.45.pth", help="model saving path")

    # Pretrain Process
    parser.add_argument("--batch_size", type=int, default=16, help="subgraph num of one batch")
    parser.add_argument("--lr", type=float, default=0.001, help="learning rate for pretraining")
    parser.add_argument("--decay", type=float, default=0.0001, help="weight decay for pretraining")
    parser.add_argument("--max_epoches", type=int, default=500, help="max epoches for pretraining")
    parser.add_argument("--loss_bias", type=float, default=10.0, help="extra loss that will be added to contrastive loss calculation")
    parser.add_argument("--adapt_step", type=int, default=2, help="model adapt steps for meta-learning")

    # Trainging enviorment
    parser.add_argument("--gpu", type=int, default=0, help="GPU id to use, -1 for CPU")
    parser.add_argument("--seed", type=int, default=40, help="random seed")
    parser.add_argument("--patience", type=int, default=20, help="early stop steps")
    
    args = parser.parse_args()
    return args


def get_downstream_args():
    parser = argparse.ArgumentParser(description='Args for Pretrain')

    # Datasets & Models
    parser.add_argument("--task", type=str, default='node', help="if node level tasks")
    parser.add_argument("--dataset", type=str, default="Coauthor_CS",help="Datasets used for downstream tasks")
    parser.add_argument("--pretrained_model", type=str, default="pretrained_gnn/Citation0.15.pth", help="pretrained model path")
    parser.add_argument("--shot", type=int, default=100, help="shot for few-shot learning")
    parser.add_argument("--k_hop", type=int, default=2, help="k-hop subgraph")
    parser.add_argument("--k_hop_nodes", type=int, default=-1, help="max nodes num for k-hop subgraph, -1 for no limiting")
    parser.add_argument("--gnn_layer", type=int, default=2, help="layer num for gnn")
    parser.add_argument("--mlp_layer", type=int, default=2, help="layer num for mlp")
    parser.add_argument("--input_dim", type=int, default=500, help="input dimension")
    parser.add_argument("--hidden_dim", type=int, default=200, help="hidden dimension")
    parser.add_argument("--output_dim", type=int, default=200, help="output dimension(also dimension of projector and answering fuction)")
    parser.add_argument("--batch_size", type=int, default=100, help="batch size for induced graphs")

    # Prompt
    parser.add_argument("--prompt_num", type=int, default=8, help="prompt num for each component")
    parser.add_argument("--prompt_dim", type=int, default=200, help="dimension of prompt, should be same as hidden_dim")
    parser.add_argument("--prompt_path", type=str, default='downstream_model/prompt_pools.pth', help="prompt pool and head saving path")
    parser.add_argument("--load_components_path", type=str, nargs='+',default=['downstream_model/prompt_pool_coauthor_cs_task0.pth',
                'downstream_model/prompt_pool_coauthor_physics_task0.pth'], help="FOR L2SPLUS: prompt components loading path")

    # Downstream Tasks
    parser.add_argument("--lr", type=float, default=0.002, help="learning rate for downstream training")
    parser.add_argument("--decay", type=float, default=0.0001, help="weight decay for downstream training")
    parser.add_argument("--max_epoches", type=int, default=600, help="max epoches for downstream training")
    parser.add_argument("--ortho_weight", type=float, default=1, help="weight for ortho regularization")
    parser.add_argument("--label_smoothing", type=float, default=0.1, help="label_smoothing for over-fitting")

    # Trainging enviorment
    parser.add_argument("--gpu", type=int, default=0, help="GPU id to use, -1 for CPU")
    parser.add_argument("--seed", type=int, default=1145, help="random seed")
    parser.add_argument("--patience", type=int, default=10, help="early stop steps")

    args = parser.parse_args()
    return args