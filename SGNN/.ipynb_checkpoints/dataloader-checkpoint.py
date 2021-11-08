from torch.utils.data import Dataset, DataLoader
from torch.utils.data import DataLoader, random_split
import torch
import torch.nn.functional as F

import dgl
import pandas as pd
import random
import numpy as np
from torch.utils.data.sampler import SubsetRandomSampler
from dgl.dataloading.pytorch import NodeDataLoader
from dgl.dataloading import MultiLayerNeighborSampler
from dgl.distributed import dist_dataloader


class SgnnDataset(Dataset):
    def __init__(self, pickle_dir, pickle_name):
        self.pickle_dir =  pickle_dir
        self.pickle_name = pickle_name
        self.graphs = []
        self.raw_data = pd.read_pickle(self.pickle_dir+self.pickle_name)  
        
        for idx, row in self.raw_data.iterrows():
            
            G = row['G']
            feature = row['feature']
            feature[:,:3] = feature[:,:3] / 255 # rgb normalization
            count = row['superpixel_num']  
            count = count  / np.argmax(count)  # n / n_max
            edges = row['edges']
            label_gt = row['label_gt']
            num_nodes = len(label_gt)
            # print(num_nodes)
            edges_src = torch.tensor(edges[:,0])
            edges_dst = torch.tensor(edges[:,1])
            dgel_graph = dgl.graph((edges_src , edges_dst), num_nodes=num_nodes, idtype=torch.int32)
            
            dgel_graph.ndata['feat'] = torch.from_numpy(feature)
            dgel_graph.ndata['pixel_num'] = torch.from_numpy(count)
            dgel_graph.ndata['label'] = torch.from_numpy(np.array(label_gt))
            dgel_graph = dgl.add_self_loop(dgel_graph)
            self.graphs.append(dgel_graph)
    
    def __getitem__(self, i):
        
        return self.graphs[i]

    def __len__(self):
        return len(self.raw_data)
        
        
def data_generator(config):
    # dataset setting
    shuffle = config['data']['shuffle']
    random_seed= 42
    shuffle_dataset = shuffle   
    validation_split = config['data']['val_holdout_frac']
    batch_size = config['training']['batch_size']
    num_class = config['training']['n_classes']
    data_dir = config['data']['pickle_dir']
    data_name = config['data']['pickle_name']
    val_name = config['data']['val_pickle_name']
    dataset_type = data_name.split('_')[0]
    sampler = config['sampler']['sampler_true']
    sampler_neighbor = config['sampler']['sampler_neighbor']
    sampler_weight = config['sampler']['sampler_weight']
    # call dataset
    print('Reading Train Dataset...')
    if dataset_type=='city':
        train_dataset = SgnnDataset(data_dir, data_name) 
        val_dataset = SgnnDataset(data_dir, val_name) 
    elif dataset_type=='uav':
        # split dataset into train/valid
        dataset = SgnnDataset(data_dir, data_name) 
        train_size = int((1-validation_split) * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])   
    print('Loading Train Dataset...')
    
    if sampler : 
        # 모든 이미지 그래프를 하나의 batch graph로
        train_dataset = dgl.batch(train_dataset)
        val_dataset = dgl.batch(val_dataset)

        # neighbor sampler (randomly select 5 neighbor nodes for all layers)
        graph_sampler = dgl.dataloading.MultiLayerNeighborSampler(sampler_neighbor)

        # Weighted Random Sampler
        train_label_weights = get_sample_weights(train_dataset.ndata["label"],sampler_weight)
        train_label_weights = torch.as_tensor(train_label_weights, dtype=torch.int32)
        train_sampler = torch.utils.data.sampler.WeightedRandomSampler(train_label_weights, len(train_label_weights), replacement=False)
        train_size_len = torch.tensor(np.arange(train_dataset.num_nodes())).to(torch.int32)
        val_size_len = torch.tensor(np.arange(val_dataset.num_nodes())).to(torch.int32)
        train_loader = dgl.dataloading.NodeDataLoader(train_dataset, train_size_len, graph_sampler, 
                                                  batch_size=batch_size, device='cpu', shuffle=True, drop_last=False)      
        val_loader = dgl.dataloading.NodeDataLoader(val_dataset, val_size_len, graph_sampler, 
                                                  batch_size=batch_size, device='cpu', shuffle= True, drop_last=False)

        return train_loader, val_loader
    
    # no sampler mode
    else : 
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn = collate)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,collate_fn = collate)

        return train_loader, val_loader

def get_sample_weights(labels, weights_list):
    class_sample_count = torch.tensor([(labels == t).sum() for t in torch.unique(labels, sorted=True)])
    # weight = 1000. / class_sample_count.float()
    weight = torch.tensor(weights_list)
    sample_weights = torch.tensor([weight[t.item()] for t in labels])
    return sample_weights

def collate(samples):
    
    batched_graph = dgl.batch(samples)
    
    return batched_graph
    
