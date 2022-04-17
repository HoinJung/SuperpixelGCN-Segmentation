import pandas as pd 
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
import numpy as np
import dgl.function as fn
from dgl.nn import SAGEConv, GatedGraphConv, ChebConv, DenseGraphConv, TAGConv, SGConv, APPNPConv, \
EdgeConv, GraphConv, RelGraphConv,GATConv, GINConv, GMMConv, AGNNConv, DotGatConv
from utils import MLPReadout, Aggregator, MeanAggregator, MaxPoolAggregator, LSTMAggregator, NodeApply

# python train.py -yml train_city.yaml
class GraphSageLayer(nn.Module):

    def __init__(self, in_feats, out_feats, activation, dropout,
                 aggregator_type, batch_norm, residual=False,
                 bias=True, dgl_builtin=False, conv_type = 'sage',tag_kernal=2,appnp_rate = 0.1, device=None):
        super().__init__()
        self.in_channels = in_feats
        self.out_channels = out_feats
        self.aggregator_type = aggregator_type
        self.batch_norm = batch_norm
        self.residual = residual
        self.dgl_builtin = dgl_builtin
        self.conv_type = conv_type
        self.device = device
        if in_feats != out_feats:
            self.residual = False
        
        self.dropout = nn.Dropout(p=dropout)
        
        if dgl_builtin == False:
            
            self.nodeapply = NodeApply(in_feats, out_feats, activation, dropout,
                                   bias=bias)
            if aggregator_type == "maxpool":
                self.aggregator = MaxPoolAggregator(in_feats, in_feats,
                                                    activation, bias)
            elif aggregator_type == "lstm":
                self.aggregator = LSTMAggregator(in_feats, in_feats)
            else:
                self.aggregator = MeanAggregator()
        else:
            
            if conv_type == 'sage':
                self.sageconv = SAGEConv(in_feats, out_feats, aggregator_type,
                        dropout, activation=activation)
            elif conv_type == 'densesage':
                self.sageconv = DenseGraphConv(in_feats, out_feats, 
                        dropout, activation=activation)
            elif conv_type == 'cheb':
                
                self.sageconv = ChebConv(in_feats, out_feats, 
                        k=2, activation=None)
            elif conv_type == 'tag':
                self.sageconv = TAGConv(in_feats, out_feats, 
                        k=tag_kernal, activation=None)
            elif conv_type == 'sg':
                self.sageconv = SGConv(in_feats, out_feats, 
                        k=2)
                
            elif conv_type == 'appnp':
                self.sageconv = APPNPConv(k=tag_kernal, alpha=appnp_rate)
            elif conv_type == 'gate':
                self.sageconv = GatedGraphConv(in_feats, out_feats, 2, 3)
            elif conv_type == 'edge':
                self.sageconv = EdgeConv(in_feats, out_feats)
            elif conv_type == 'graph':
                self.sageconv = GraphConv(in_feats, out_feats,activation=activation)
            elif conv_type == 'rel' :
                self.sageconv = RelGraphConv(in_feats, out_feats,num_rels=3,dropout=dropout,activation=activation)
            elif conv_type == 'gat' :
                self.sageconv = GATConv(in_feats, out_feats, num_heads=256, feat_drop=0.0, attn_drop=0.0, negative_slope=0.2, activation=activation)
            elif conv_type == 'gmm' :
                self.sageconv = GMMConv(in_feats, out_feats, dim = 5, n_kernels = 2, aggregator_type = aggregator_type)
            elif conv_type == 'gin' :
                lin = nn.Linear(in_feats, out_feats)
                self.sageconv = GINConv(lin, 'mean')
            elif conv_type == 'agnn' :
                self.sageconv = AGNNConv()
            elif conv_type == 'dotgat':
                self.sageconv = DotGatConv(in_feats, out_feats, num_heads=in_feats)
                
        
        if self.batch_norm == 'bn':
            self.batchnorm_h = nn.BatchNorm1d(out_feats)
        elif self.batch_norm == 'gn': # groupnorm
            self.batchnorm_h = Norm('gn',hidden_dim=out_feats,device = self.device)
      
            

    def forward(self, g, h):
        h_in = h              # for residual connection
        
        if self.dgl_builtin == False:
            h = self.dropout(h)
            g.ndata['h'] = h
            #g.update_all(fn.copy_src(src='h', out='m'), 
            #             self.aggregator,
            #             self.nodeapply)
            if self.aggregator_type == 'maxpool':
                g.ndata['h'] = self.aggregator.linear(g.ndata['h'])
                g.ndata['h'] = self.aggregator.activation(g.ndata['h'])
                g.update_all(fn.copy_src('h', 'm'), fn.max('m', 'c'), self.nodeapply)
            elif self.aggregator_type == 'lstm':
                g.update_all(fn.copy_src(src='h', out='m'), 
                             self.aggregator,
                             self.nodeapply)
            else:
                # copy_src: 'h'feature를 전달하여 source vertex의 'm'ailbox에 저장. mean: 'm'의 mean을 도착 노드의 c에 저장. 
                g.update_all(fn.copy_src('h', 'm'), fn.mean('m', 'c'), self.nodeapply)
            h = g.ndata['h']
        else:
            h = self.sageconv(g, h)
            
            

        if self.batch_norm == 'bn':
            h = self.batchnorm_h(h)
        elif self.batch_norm == 'gn':
            h = self.batchnorm_h(g, h)
        h = F.relu(h)
        if self.residual:
            h = h_in + h       # residual connection
        
        return h
    
    def __repr__(self):
        return '{}(in_channels={}, out_channels={}, aggregator={}, residual={})'.format(self.__class__.__name__,
                                              self.in_channels,
                                              self.out_channels, self.aggregator_type, self.residual)

    
class GraphSageNet(nn.Module):
    """
    Grahpsage network with multiple GraphSageLayer layers
    """
    def __init__(self, net_params):
        super().__init__()
        in_dim = net_params['in_dim']
        hidden_dim = net_params['hidden_dim']
        out_dim = net_params['out_dim']
        self.n_classes = net_params['training']['n_classes']
        in_feat_dropout = net_params['in_feat_dropout'] 
        dropout = net_params['dropout']
        aggregator_type = net_params['sage_aggregator']
        n_layers = net_params['Layer']   # number of GraphSageLayer. 몇 개의 hop을 볼지. 
        batch_norm = net_params['batch_norm']
        residual = net_params['residual']
        dgl_builtin = net_params['dgl_builtin']
        tag_kernal = net_params['tag_kernal']
        appnp_rate = net_params['appnp_rate']
        self.conv_type = net_params['conv_type']
        self.readout = net_params['readout']
        self.device = net_params['device']
        self.embedding_h = nn.Linear(in_dim, hidden_dim)
        self.in_feat_dropout = nn.Dropout(in_feat_dropout)

        self.layers = nn.ModuleList([GraphSageLayer(hidden_dim, hidden_dim, F.relu,
                                              dropout, aggregator_type, batch_norm, residual, device = self.device, dgl_builtin =  dgl_builtin, conv_type = self.conv_type, tag_kernal = tag_kernal) for _ in range(n_layers-1)])
        self.layers.append(GraphSageLayer(hidden_dim, out_dim, F.relu, dropout, aggregator_type, batch_norm, residual,device = self.device,  dgl_builtin =  dgl_builtin, conv_type = self.conv_type, tag_kernal = tag_kernal))
        self.MLP_layer = MLPReadout(out_dim, self.n_classes) # readout layer 수정. hidden_dim=out_dim=108.
        
    def forward(self, g, h):
        # print("Work!")
        h = self.embedding_h(h)
        h = self.in_feat_dropout(h)
        for conv in self.layers:
            h = conv(g, h) # [n_nodes,out_dim]
            
        h_out = self.MLP_layer(h)
        
        return h_out # [n_nodes,n_classes]

# Groupnorm
class Norm(nn.Module):

    def __init__(self, norm_type, hidden_dim=300, device=None, print_info=None):
        super(Norm, self).__init__()
        assert norm_type in ['bn', 'gn', None]
        self.norm = None
        self.print_info = print_info
        self.device = device
        # self.device = 'cpu'
        if norm_type == 'bn':
            self.norm = nn.BatchNorm1d(hidden_dim)
        elif norm_type == 'gn':
            self.norm = norm_type
            self.weight = nn.Parameter(torch.ones(hidden_dim))
            self.bias = nn.Parameter(torch.zeros(hidden_dim))

    def forward(self, graph, tensor, print_=False):
        if self.norm is not None and type(self.norm) != str:
            return self.norm(tensor)
        elif self.norm is None:
            return tensor
        batch_list = graph.batch_num_nodes()
        batch_size = len(batch_list)
        batch_list = torch.tensor(batch_list,dtype=torch.long).to(self.device)
        batch_index = torch.arange(batch_size).to(self.device).repeat_interleave(batch_list)
        batch_index = batch_index.view((-1,) + (1,) * (tensor.dim() - 1)).expand_as(tensor)
        mean = torch.zeros(batch_size, *tensor.shape[1:]).to(self.device)
        mean = mean.scatter_add_(0, batch_index, tensor)
        mean = (mean.T / batch_list).T
        mean = mean.repeat_interleave(batch_list, dim=0)
        sub = tensor - mean
        std = torch.zeros(batch_size, *tensor.shape[1:]).to(self.device)
        std = std.scatter_add_(0, batch_index, sub.pow(2))
        std = ((std.T / batch_list).T + 1e-6).sqrt()
        std = std.repeat_interleave(batch_list, dim=0)
        # return sub / std
        return self.weight * sub / std + self.bias    
    
