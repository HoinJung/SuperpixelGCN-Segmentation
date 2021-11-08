import pandas as pd 
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
import numpy as np
import dgl.function as fn
from dgl.nn import SAGEConv

class MLPReadout(nn.Module):

    def __init__(self, input_dim, output_dim, L=2): #L=nb_hidden_layers
        super().__init__()
        list_FC_layers = [ nn.Linear( input_dim//2**l , input_dim//2**(l+1) , bias=True ) for l in range(L) ]
        list_FC_layers.append(nn.Linear( input_dim//2**L , output_dim , bias=True ))
        self.FC_layers = nn.ModuleList(list_FC_layers)
        self.L = L
        
    def forward(self, x):
        y = x
        for l in range(self.L):
            y = self.FC_layers[l](y)
            y = F.relu(y)
        y = self.FC_layers[self.L](y)
        return y
    
    
class GraphSageLayer(nn.Module):

    def __init__(self, in_feats, out_feats, activation, dropout,
                 aggregator_type, batch_norm, residual=False, 
                 bias=True, dgl_builtin=False):
        super().__init__()
        self.in_channels = in_feats
        self.out_channels = out_feats
        self.aggregator_type = aggregator_type
        self.batch_norm = batch_norm
        self.residual = residual
        self.dgl_builtin = dgl_builtin
        
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
            self.sageconv = SAGEConv(in_feats, out_feats, aggregator_type,
                    dropout, activation=activation)
        
        if self.batch_norm:
            self.batchnorm_h = nn.BatchNorm1d(out_feats)

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

        if self.batch_norm:
            h = self.batchnorm_h(h)
        
        if self.residual:
            h = h_in + h       # residual connection
        
        return h
    
    def __repr__(self):
        return '{}(in_channels={}, out_channels={}, aggregator={}, residual={})'.format(self.__class__.__name__,
                                              self.in_channels,
                                              self.out_channels, self.aggregator_type, self.residual)

    
  
"""
    Aggregators for GraphSage
"""
class Aggregator(nn.Module):
    """
    Base Aggregator class. 
    """

    def __init__(self):
        super().__init__()

    def forward(self, node):
        neighbour = node.mailbox['m']
        c = self.aggre(neighbour)
        return {"c": c}

    def aggre(self, neighbour):
        # N x F
        raise NotImplementedError


class MeanAggregator(Aggregator):
    """
    Mean Aggregator for graphsage
    """

    def __init__(self):
        super().__init__()

    def aggre(self, neighbour):
        mean_neighbour = torch.mean(neighbour, dim=1)
        return mean_neighbour


class MaxPoolAggregator(Aggregator):
    """
    Maxpooling aggregator for graphsage
    """

    def __init__(self, in_feats, out_feats, activation, bias):
        super().__init__()
        self.linear = nn.Linear(in_feats, out_feats, bias=bias)
        self.activation = activation

    def aggre(self, neighbour):
        neighbour = self.linear(neighbour)
        if self.activation:
            neighbour = self.activation(neighbour)
        maxpool_neighbour = torch.max(neighbour, dim=1)[0]
        return maxpool_neighbour


class LSTMAggregator(Aggregator):
    """
    LSTM aggregator for graphsage
    """

    def __init__(self, in_feats, hidden_feats):
        super().__init__()
        self.lstm = nn.LSTM(in_feats, hidden_feats, batch_first=True)
        self.hidden_dim = hidden_feats
        self.hidden = self.init_hidden()

        nn.init.xavier_uniform_(self.lstm.weight,
                                gain=nn.init.calculate_gain('relu'))

    def init_hidden(self):
        """
        Defaulted to initialite all zero
        """
        return (torch.zeros(1, 1, self.hidden_dim),
                torch.zeros(1, 1, self.hidden_dim))

    def aggre(self, neighbours):
        """
        aggregation function
        """
        # N X F
        rand_order = torch.randperm(neighbours.size()[1])
        neighbours = neighbours[:, rand_order, :]

        (lstm_out, self.hidden) = self.lstm(neighbours.view(neighbours.size()[0], neighbours.size()[1], -1))
        return lstm_out[:, -1, :]

    def forward(self, node):
        neighbour = node.mailbox['m']
        c = self.aggre(neighbour)
        return {"c": c}
    
class NodeApply(nn.Module):
    """
    Works -> the node_apply function in DGL paradigm
    """

    def __init__(self, in_feats, out_feats, activation, dropout, bias=True):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.linear = nn.Linear(in_feats * 2, out_feats, bias)
        self.activation = activation

    def concat(self, h, aggre_result):
        bundle = torch.cat((h, aggre_result), 1)
        bundle = self.linear(bundle)
        return bundle

    def forward(self, node):
        h = node.data['h']
        c = node.data['c']
        bundle = self.concat(h, c) # 본인 node feature랑 agrregate된 feature랑 concat해서 linear function을 거침. 
        bundle = F.normalize(bundle, p=2, dim=1)
        if self.activation:
            bundle = self.activation(bundle)
        return {"h": bundle}
    
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
        self.readout = net_params['readout']
        self.device = net_params['device']
        self.embedding_h = nn.Linear(in_dim, hidden_dim)
        self.in_feat_dropout = nn.Dropout(in_feat_dropout)
        
        self.layers = nn.ModuleList([GraphSageLayer(hidden_dim, hidden_dim, F.relu,
                                              dropout, aggregator_type, batch_norm, residual) for _ in range(n_layers-1)])
        self.layers.append(GraphSageLayer(hidden_dim, out_dim, F.relu, dropout, aggregator_type, batch_norm, residual))
        self.MLP_layer = MLPReadout(out_dim, self.n_classes) # readout layer 수정. hidden_dim=out_dim=108.
        
    def forward(self, g, h):
        h = self.embedding_h(h)
        h = self.in_feat_dropout(h)
        for conv in self.layers:
            h = conv(g, h) # [n_nodes,out_dim]
            
        h_out = self.MLP_layer(h)
        
        # h_out = F.sigmoid(h_out)
        return h_out # [n_nodes,n_classes]
    
class GraphSageNet_sampler(nn.Module):
    def __init__(self, net_params):
        super().__init__()
        in_dim = net_params['in_dim']
        hidden_dim = net_params['hidden_dim']
        out_dim = net_params['out_dim']
        self.n_classes = net_params['training']['n_classes']
        in_feat_dropout = net_params['in_feat_dropout'] 
        dropout = net_params['dropout']
        aggregator_type = net_params['sage_aggregator']
        self.n_layers = net_params['Layer']
        self.batch_norm = net_params['batch_norm']
        residual = net_params['residual']
        self.device = net_params['device']
        norm = None
        bias = True
        activation = F.relu
        
        self.readout = net_params['readout']
        
        self.embedding_h = nn.Linear(in_dim, hidden_dim)
        self.layers = nn.ModuleList([SAGEConv(hidden_dim, hidden_dim, aggregator_type, dropout, bias, norm) for i in range(self.n_layers)])
        self.batchnorms = nn.ModuleList([ nn.BatchNorm1d(hidden_dim) for i in range(self.n_layers)])
        self.MLP_layer = MLPReadout(out_dim, self.n_classes)
        self.relu = F.relu
        
    def forward(self, blocks, h):
        h = self.embedding_h(h)
        
        for i in range(self.n_layers):
            h_dst = h[:blocks[i].num_dst_nodes()]
            h = self.batchnorms[i](self.layers[i](blocks[i], (h, h_dst)))
            
            if i != self.n_layers-1 :
                h = self.relu(h)
      
        h_out = self.MLP_layer(h)
        
        return h_out
