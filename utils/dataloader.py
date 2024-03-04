import os
import dgl
import torch
import random
import chardet
import numpy as np
import pandas as pd
from tqdm import tqdm
from torch_geometric.data import Data
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from torch_geometric.transforms import SVDFeatureReduction
from torch_geometric.datasets import Planetoid, Amazon, Yelp
import torch.nn.functional as F


def csv_reader(data_path):
    with open(data_path, 'rb') as f:
        result = chardet.detect(f.read())
    encoding = result['encoding']
    data = pd.read_csv(os.path.join(data_path), encoding=encoding)
    return data

def pretrain_dataloader(input_dim:int, dataset:str):
    
    if dataset == 'Yelp_Fraud' or dataset == 'Amazon_Fraud':
        if dataset == 'Yelp_Fraud':
            data = dgl.data.FraudDataset('yelp')
            dataname = 'Yelp_Fraud'
        else:
            data = dgl.data.FraudDataset('amazon')
            dataname = 'Amazon_Fraud'
        num_classes = 2
        g = data[0]
        g = dgl.to_homogeneous(g, ndata=['feature','label','train_mask','val_mask','test_mask'])
        src, dst = g.edges()
        edge_index = torch.stack([src, dst], dim=0)
        x = g.ndata['feature']
        y = g.ndata['label']
        data = Data(x=x, edge_index=edge_index, y=y)

    elif dataset == 'S-FFSD':
        num_classes = 2
        if os.path.exists('data/S-FFSD'):
            g = dgl.load_graphs(f'data/S-FFSD/S-FFSD_graph.bin')[0][0]
        else:
            data = csv_reader('data/S-FFSD/S-FFSD_feat.csv')
            data = data.fillna(0)
            cal_list = ["Source", "Target", "Location", "Type"]
            for col in cal_list:
                le = LabelEncoder()
                data[col] = le.fit_transform(data[col].apply(str).values)

            feat_data = data.drop("Labels", axis=1)
            labels = data["Labels"]
            # Connect Points
            etypes = ["Source", "Target", "Location", "Type"]
            graph_dict = {}
            for etype in etypes:
                src, tgt = [], []
                edge_per_trans = 5
                for _, c_df in tqdm(data.groupby(etype), desc=etype):
                    c_df = c_df.sort_values(by="Time")
                    df_len = len(c_df)
                    sorted_idxs = c_df.index
                    c_src = [sorted_idxs[i] for i in range(df_len) for j in range(edge_per_trans) if i + j < df_len]
                    c_tgt = [sorted_idxs[i + j] for i in range(df_len) for j in range(edge_per_trans) if i + j < df_len]
                    src.extend(c_src)
                    tgt.extend(c_tgt)
                src = np.array(src)
                tgt = np.array(tgt)
                graph_dict[('trans', etype, 'trans')] = (src, tgt)
            g = dgl.heterograph(graph_dict)
            scaler = StandardScaler()

            g.nodes['trans'].data['label'] = torch.from_numpy(labels.to_numpy())
            g.nodes['trans'].data['feature'] = torch.from_numpy(scaler.fit_transform(feat_data))

            g = dgl.to_homogeneous(g,ndata=['feature','label'])

            g_path = f'data/S-FFSD/S-FFSD_graph.bin'
            dgl.data.utils.save_graphs(g_path, [g])
        src, dst = g.edges()
        edge_index = torch.stack([src, dst], dim=0)
        x = g.ndata['feature']
        y = g.ndata['label']
        data = Data(x=x, edge_index=edge_index, y=y)
        dataname = 'S-FFSD'

    elif dataset == 'Amazon_Photo' or dataset == 'Amazon_Computer':
        if dataset == 'Amazon_Photo':
            dataset = Amazon(root='data/', name='photo')
            dataname = 'Photo'
        elif dataset == 'Amazon_Computer':
            dataset = Amazon(root='data/', name='computers')
            dataname = 'Computers'
        num_classes = dataset.num_classes
        data = dataset.data

    elif dataset == 'Cora' or dataset == 'CiteSeer' or dataset == 'PubMed':
        dataset = Planetoid(root='dataset/', name=dataset)
        data = dataset.data
        num_classes = dataset.num_classes

    if data.x.shape[1] < input_dim:
        padding_size = input_dim - data.x.shape[1]
        data.x = F.pad(data.x, (0, padding_size), 'constant', 0)
    else:
        svd_reduction = SVDFeatureReduction(out_channels=input_dim) # use SVD to uniform features dimension
        data = svd_reduction(data)

    return data, dataname, num_classes
