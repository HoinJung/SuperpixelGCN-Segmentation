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
from dgl.distributed import DistDataLoader
from dgl.data.utils import load_graphs


class SgnnDataset(Dataset):
    def __init__(self, pickle_dir, pickle_name):
        self.pickle_dir =  pickle_dir
        self.pickle_name = pickle_name
        self.graphs = []
        
        
        glist, label_dict = load_graphs(self.pickle_dir+self.pickle_name.replace('pickle','bin'))    
        self.graphs=glist
        
    def __getitem__(self, i):
        
        return self.graphs[i]

    def __len__(self):
        return len(self.graphs)
        
        
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
    
    if dataset_type=='uav':
        # split dataset into train/valid
        dataset = SgnnDataset(data_dir, data_name) 
        train_size = int((1-validation_split) * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
        
    else :
        train_dataset = SgnnDataset(data_dir, data_name) 
        val_dataset = SgnnDataset(data_dir, val_name) 
    
    print('Loading Train Dataset...')
    
    if sampler : 
        train_dataset = dgl.batch(train_dataset)
        val_dataset = dgl.batch(val_dataset)

        # neighbor sampler
        graph_sampler = dgl.dataloading.MultiLayerNeighborSampler(sampler_neighbor)

        # Weighted Random Sampler
        train_label_weights = get_sample_weights(train_dataset.ndata["label"],sampler_weight)
        train_label_weights = torch.as_tensor(train_label_weights, dtype=torch.int32)
        train_sampler = torch.utils.data.sampler.WeightedRandomSampler(train_label_weights, len(train_label_weights), replacement=False)
        train_size_len = torch.tensor(np.arange(train_dataset.num_nodes())).to(torch.int32)
        val_size_len = torch.tensor(np.arange(val_dataset.num_nodes())).to(torch.int32)
        train_loader = dgl.dataloading.NodeDataLoader(train_dataset, train_size_len, graph_sampler, sampler=train_sampler,
                                                  batch_size=batch_size, device='cpu', shuffle=False, drop_last=False)      
        val_loader = dgl.dataloading.NodeDataLoader(val_dataset, val_size_len, graph_sampler, 
                                                  batch_size=batch_size, device='cpu', shuffle=True, drop_last=False)

        return train_loader, val_loader
    
    # no sampler mode
    else : 
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn = collate)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,collate_fn = collate)

        return train_loader, val_loader

def get_sample_weights(labels, weights_list):
    
    weight = torch.tensor(weights_list)
    sample_weights = torch.tensor([weight[t.item()] for t in labels])
    return sample_weights

def collate(samples):
    
    batched_graph = dgl.batch(samples)
    
    return batched_graph
    
