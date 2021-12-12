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
                 bias=True, dgl_builtin=False, conv_type = 'sage',tag_kernal=2):
        super().__init__()
        self.in_channels = in_feats
        self.out_channels = out_feats
        self.aggregator_type = aggregator_type
        self.batch_norm = batch_norm
        self.residual = residual
        self.dgl_builtin = dgl_builtin
        self.conv_type = conv_type
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
                self.sageconv = APPNPConv(k=3, alpha=0.5)
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
        self.conv_type = net_params['conv_type']
        self.readout = net_params['readout']
        self.device = net_params['device']
        self.embedding_h = nn.Linear(in_dim, hidden_dim)
        self.in_feat_dropout = nn.Dropout(in_feat_dropout)

        self.layers = nn.ModuleList([GraphSageLayer(hidden_dim, hidden_dim, F.relu,
                                              dropout, aggregator_type, batch_norm, residual, dgl_builtin =  dgl_builtin, conv_type = self.conv_type, tag_kernal = tag_kernal) for _ in range(n_layers-1)])
        self.layers.append(GraphSageLayer(hidden_dim, out_dim, F.relu, dropout, aggregator_type, batch_norm, residual, dgl_builtin =  dgl_builtin, conv_type = self.conv_type, tag_kernal = tag_kernal))
        self.MLP_layer = MLPReadout(out_dim, self.n_classes) # readout layer 수정. hidden_dim=out_dim=108.
        
    def forward(self, g, h):
        # print("Work!")
        h = self.embedding_h(h)
        h = self.in_feat_dropout(h)
        for conv in self.layers:
            h = conv(g, h) # [n_nodes,out_dim]
            
        h_out = self.MLP_layer(h)
        
        return h_out # [n_nodes,n_classes]
class GNN(nn.Module):
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
        self.conv_type = net_params['conv_type']
        self.readout = net_params['readout']
        self.device = net_params['device']
        self.embedding_h = nn.Linear(in_dim, hidden_dim)
        self.in_feat_dropout = nn.Dropout(in_feat_dropout)

        self.layers = nn.ModuleList([GraphSageLayer(hidden_dim, hidden_dim, F.relu,
                                              dropout, aggregator_type, batch_norm, residual, dgl_builtin =  dgl_builtin, conv_type = self.conv_type, tag_kernal = tag_kernal) for _ in range(n_layers-1)])
        self.layers.append(GraphSageLayer(hidden_dim, out_dim, F.relu, dropout, aggregator_type, batch_norm, residual, dgl_builtin =  dgl_builtin, conv_type = self.conv_type, tag_kernal = tag_kernal))
        self.MLP_layer = MLPReadout(out_dim, self.n_classes) # readout layer 수정. hidden_dim=out_dim=108.
        
    def forward(self, g, h):
        # print("Work!")
        h = self.embedding_h(h)
        h = self.in_feat_dropout(h)
        for conv in self.layers:
            h = conv(g, h) # [n_nodes,out_dim]
            
        h_out = self.MLP_layer(h)
        
        return h_out # [n_nodes,n_classes]
    
class GraphMultiNet(nn.Module):
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
        self.conv_type = net_params['conv_type']
        self.readout = net_params['readout']
        self.device = net_params['device']
        
        # multi scale
        self.layer_1 = net_params['multi_scale_mode']['layer_scale_1']
        self.layer_2 = net_params['multi_scale_mode']['layer_scale_2']
        self.layer_3 = net_params['multi_scale_mode']['layer_scale_3']
        
        self.embedding_h = nn.Linear(in_dim, hidden_dim)
        self.in_feat_dropout = nn.Dropout(in_feat_dropout)
        self.layers = nn.ModuleList([GraphSageLayer(hidden_dim, hidden_dim, F.relu,
                                              dropout, aggregator_type, batch_norm, residual, dgl_builtin =  dgl_builtin, conv_type = self.conv_type, tag_kernal = tag_kernal) for _ in range(n_layers-1)])
        self.layers.append(GraphSageLayer(hidden_dim, out_dim, F.relu, dropout, aggregator_type, batch_norm, residual, dgl_builtin =  dgl_builtin, conv_type = self.conv_type, tag_kernal = tag_kernal))
        
#         self.layers1 = nn.ModuleList([GraphSageLayer(hidden_dim, hidden_dim, F.relu,\
#                                      dropout, aggregator_type, batch_norm, \
#                                     residual, dgl_builtin =  dgl_builtin, \
#                                     conv_type = self.conv_type, tag_kernal = tag_kernal) \
#                                      for _ in range(self.layer_1-1)])
        
#         self.layers2 = nn.ModuleList([GraphSageLayer(hidden_dim, hidden_dim, F.relu,\
#                                      dropout, aggregator_type, batch_norm, \
#                                     residual, dgl_builtin =  dgl_builtin, \
#                                     conv_type = self.conv_type, tag_kernal = tag_kernal) \
#                                      for _ in range(self.layer_2-1)])
        
#         self.layers3 = nn.ModuleList([GraphSageLayer(hidden_dim, hidden_dim, F.relu,\
#                                      dropout, aggregator_type, batch_norm, \
#                                     residual, dgl_builtin =  dgl_builtin, \
#                                     conv_type = self.conv_type, tag_kernal = tag_kernal) \
#                                      for _ in range(self.layer_3-1)])
        
        
#         self.layers1.append(GraphSageLayer(hidden_dim, out_dim, F.relu, dropout, aggregator_type, batch_norm, residual, dgl_builtin =  dgl_builtin, conv_type = self.conv_type, tag_kernal = tag_kernal))
#         self.layers2.append(GraphSageLayer(hidden_dim, out_dim, F.relu, dropout, aggregator_type, batch_norm, residual, dgl_builtin =  dgl_builtin, conv_type = self.conv_type, tag_kernal = tag_kernal))
#         self.layers3.append(GraphSageLayer(hidden_dim, out_dim, F.relu, dropout, aggregator_type, batch_norm, residual, dgl_builtin =  dgl_builtin, conv_type = self.conv_type, tag_kernal = tag_kernal))
        # self.MLP_layer = MLPReadout(out_dim, self.n_classes) # readout layer 수정. hidden_dim=out_dim=108.
        # self.MLP_layer_0_1 = MLPReadout(hidden_dim, out_dim) # readout layer 수정. hidden_dim=out_dim=108.
        # self.MLP_layer_0_2 = MLPReadout(hidden_dim, out_dim)
        # self.MLP_layer_0_3 = MLPReadout(hidden_dim, out_dim)
        # self.relu = nn.ReLU()
        # self.bn1 = nn.BatchNorm1d(out_dim)
        # self.bn2 = nn.BatchNorm1d(out_dim)
        # self.bn3 = nn.BatchNorm1d(out_dim)
        self.MLP_layer_1 = MLPReadout(out_dim, self.n_classes) # readout layer 수정. hidden_dim=out_dim=108.
        self.MLP_layer_2 = MLPReadout(out_dim, self.n_classes) # readout layer 수정. hidden_dim=out_dim=108.
        self.MLP_layer_3 = MLPReadout(out_dim, self.n_classes) # readout layer 수정. hidden_dim=out_dim=108.
        
        
    def forward(self, g, h):
        
        h = self.embedding_h(h)
        h = self.in_feat_dropout(h)
        h1 = self.layers[0](g,h)
        h2 = self.layers[1](g,h1)
        h3 = self.layers[2](g,h2)
        h4 = self.layers[3](g,h3)
        h5 = self.layers[4](g,h4)
        h6 = self.layers[5](g,h5)
        h7 = self.layers[6](g,h6)
        h8 = self.layers[7](g,h7)
        h9 = self.layers[8](g,h8)
        h10 = self.layers[9](g,h9)
        
        h11 = self.layers[10](g,h10)
        h12 = self.layers[11](g,h11)
        # mIoU 0.489 #256
        # h_out_1=self.MLP_layer(h4)
        # h_out_2=self.MLP_layer(h8)
        # h_out_3=self.MLP_layer(h12)
        
        # mIoU 0.494 #256
        # h_out_1=self.MLP_layer_1(h4)
        # h_out_2=self.MLP_layer_1(h8)
        # h_out_3=self.MLP_layer_1(h12)   
        # # mIoU 0.493 #256
        h_out_1=self.MLP_layer_1(h4)
        h_out_2=self.MLP_layer_2(h8)
        h_out_3=self.MLP_layer_3(h12)
        
        # mIoU 0.4922 #256
#         h_out_1=self.MLP_layer_1(h3)
#         h_out_2=self.MLP_layer_1(h6)
#         h_out_3=self.MLP_layer_2(h9)
#         h_out_4=self.MLP_layer_2(h12) 
        #1024 #0.489
#         h_out_1=self.MLP_layer_0_1(h4)
#         h_out_2=self.MLP_layer_0_2(h8)
#         h_out_3=self.MLP_layer_0_3(h12)
        
#         # h_out_4=self.MLP_layer(h12)
#         h_out_1=self.MLP_layer_1(self.bn1(self.relu(h_out_1)))
#         h_out_2=self.MLP_layer_2(self.bn2(self.relu(h_out_2)))
#         h_out_3=self.MLP_layer_3(self.bn3(self.relu(h_out_3)))
        # h_out_4=self.MLP_layer_1(h_out_4)
        # h_out_4=self.MLP_layer_1(h12)
        
        # mIoU 0.47
#         h_out_1=self.MLP_layer_1(h3)
#         h_out_2=self.MLP_layer_1(h6)
#         h_out_3=self.MLP_layer_1(h10)
        
        
        return h_out_1, h_out_2, h_out_3
        
        # h1 = torch.tensor(h)
        # h2 = torch.tensor(h)
        # h3 = torch.tensor(h)
        # for conv in self.layers1:
        #     h1 = conv(g, h1) # [n_nodes,out_dim]
        # for conv in self.layers2:
        #     h2 = conv(g, h2) # [n_nodes,out_dim]
        # for conv in self.layers3:
        #     h3 = conv(g, h3) # [n_nodes,out_dim]
        # h_concat = torch.cat([h1, h2, h3], axis=1)
        # # h_concat = torch.mean([h1, h2, h3], axis=1)
        # h_out = self.MLP_layer(h_concat)
        # return h_out # [n_nodes,n_classes]
    
    
    
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
        dgl_builtin = net_params['dgl_builtin']
        tag_kernal = net_params['tag_kernal']
        self.conv_type = net_params['conv_type']
        self.readout = net_params['readout']
        
        residual = net_params['residual']
        self.device = net_params['device']
        norm = None
        bias = True
        activation = F.relu
        
        self.readout = net_params['readout']
        
        self.embedding_h = nn.Linear(in_dim, hidden_dim)
        # self.layers = nn.ModuleList([SAGEConv(hidden_dim, hidden_dim, aggregator_type, dropout, bias, norm) for i in range(self.n_layers)])
        self.layers = nn.ModuleList([GraphSageLayer(hidden_dim, hidden_dim, F.relu,
                                              dropout, aggregator_type, self.batch_norm, residual, dgl_builtin =  dgl_builtin, conv_type = self.conv_type, tag_kernal = tag_kernal) for _ in range(self.n_layers-1)])
        self.layers.append(GraphSageLayer(hidden_dim, out_dim, F.relu, dropout, aggregator_type, self.batch_norm, residual, dgl_builtin =  dgl_builtin, conv_type = self.conv_type, tag_kernal = tag_kernal))
        
        self.MLP_layer = MLPReadout(out_dim, self.n_classes)
        self.relu = F.relu
        
    def forward(self, blocks, h):
        h = self.embedding_h(h)
        
        for i in range(self.n_layers):
            h_dst = h[:blocks[i].num_dst_nodes()]
            h = self.layers[i](blocks[i], (h, h_dst))

            # h_dst = h[:blocks[i].num_dst_nodes()]
            # h = self.layers[i](blocks[i].num_dst_nodes(), h_dst)
            
            
            # h = self.batchnorms[i](self.layers[i](blocks[i], (h, h_dst)))
            
            # if i != self.n_layers-1 :
            #     h = self.relu(h)
      
        h_out = self.MLP_layer(h)
        
        return h_out

    
    
