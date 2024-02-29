import argparse

def get_pretrain_args():
    parser = argparse.ArgumentParser(description='Args for Pretrain')
    parser.add_argument("--dataset", type=str, nargs='+', default=["Amazon_photo", "Amazon_Computer", "Amazon_Fraud"],
                        help="Datasets used for pretrain")
    parser.add_argument("--gpu", type=int, default=-1, help="GPU id to use, -1 for CPU")
    
    args = parser.parse_args()
    return args


