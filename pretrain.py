import argparse
from utils.args import get_pretrain_args
from utils.dataloader import pretrain_dataloader

if __name__ == "__main__":
    args = get_pretrain_args()

    # Get pretrain datasets
    g_list = []
    for dataset in args.dataset:
        g_list.append(pretrain_dataloader(input_dim=args.input_dim, dataset=dataset))
    print(g_list)
    