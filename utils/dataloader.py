import os
import dgl
import torch
import random
import chardet
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.manifold import SpectralEmbedding
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def csv_reader(data_path):
    with open(data_path, 'rb') as f:
        result = chardet.detect(f.read())
    encoding = result['encoding']
    data = pd.read_csv(os.path.join(data_path), encoding=encoding)
    return data

def pretrain_dataloader(input_dim:int, dataset:str):
    print("---Processing " + dataset + "---")
    if dataset == 'Yelp_Fraud' or dataset == 'Amazon_Fraud':
        if dataset == 'Yelp_Fraud':
            data = dgl.data.FraudDataset('yelp')
        else:
            data = dgl.data.FraudDataset('amazon')
        g = data[0]
        g = dgl.to_homogeneous(g,ndata=['feature','label','train_mask','val_mask','test_mask'])

    elif dataset == 'S-FFSD':
        if not os.path.exists('data/S-FFSD_graph.bin'):
            g = dgl.load_graphs(f'data/S-FFSD_graph.bin')[0][0]
        else:
            df = csv_reader(dataset)
            df = df.loc[:, ~df.columns.str.contains('Unnamed')]
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

            g_path = f'data/S-FFSD_graph.bin'
            dgl.data.utils.save_graphs(g_path, [g])

    elif dataset == 'Amazon_Photo' or dataset == 'Amazon_Computer':
        if dataset == 'Amazon_Photo':
            data = dgl.data.AmazonCoBuyPhotoDataset()
        elif dataset == 'Amazon_Computer':
            data = dgl.data.AmazonCoBuyComputerDataset()
        g = data[0]
        g.ndata['feature'] = g.ndata['feat']  # just for uniform variable name
        del g.ndata['feat']
        labels = g.ndata['label']

    embedding = SpectralEmbedding(n_components=input_dim) # use Laplace to uniform features dimension
    features = g.ndata['feature']
    features = embedding.fit_transform(features)
    g.ndata['feature'] = torch.from_numpy(features).float()
