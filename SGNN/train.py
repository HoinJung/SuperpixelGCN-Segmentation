import pandas as pd 
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import dgl
from GraphSage import GraphSageNet
import easydict 
import time
from tqdm import tqdm
import os
import random
from dataloader import SgnnDataset, data_generator
import warnings
import wandb
import time
import argparse
# from loss import select_loss
import losses
import yaml

class Trainer(object):
    def __init__(self, config):
        
        # make self config
        
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        config['device'] = self.device
        self.lr = config['training']['init_lr']
        self.optim = config['training']['optim']
        self.batch_size = config['training']['batch_size']
        self.patience = config['training']['patience']
        self.epoch = config['training']['epochs']
        self.save_dir = config['data']['checkpoint_dir']; os.makedirs(self.save_dir, exist_ok = True)
        self.n_class = config['training']['n_classes']
        self.sampler = config['sampler']['sampler_true']
        self.save_best_epoch_pth = config['training']['save_best_epoch_pth']
        self.eval_interval = config['training']['eval_interval']
       
        # multi_scale mode
        self.multi_scale = config['multi_scale_mode']['use_multi_scale']
        
            
        # code for wandb    
        self.wandb = config['wandb']
        self.wandb_proj_name = config['wandb_proj_name']
        
        # make models and data 
        self.train_data , self.valid_data = data_generator(config)
        
        # select loss function
        self.loss_name = config['loss']
        selected_loss = losses.CustomLoss(self.loss_name)
        self.loss = selected_loss.select_loss()
        # self.loss = nn.CrossEntropyLoss()
      
        # use sampler or not
        if self.sampler :
            # sampler mode
            self.model = GraphSageNet_sampler(config).to(self.device)
        elif self.multi_scale : 
            self.model = GraphMultiNet(config).to(self.device)
        elif config['test_mode']:
            self.model = GNN(config).to(self.device)
        else : 
            # graph(?) mode
            self.model = GraphSageNet(config).to(self.device)
            
        
        # if you want to use multi-GPU
        # self.model = torch.nn.DataParallel(self.model)       
        
        # select optimizer
        if self.optim == 'Adam' : 
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        elif self.optim == 'SGD' : 
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr)
            
        # select lr_scheduler
        self.lr_scheduler = config['learning_rate']['lr_scheduler']
        if config['learning_rate']['scheduler']:
            if self.lr_scheduler =='cosine':
                self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=self.optimizer, 
                                                                            T_max=config['learning_rate']['T_max'], 
                                                                            eta_min=config['learning_rate']['eta_min'], 
                                                                            )
            elif self.lr_scheduler =='step':
                self.scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer=self.optimizer,
                                                                      milestones=config['learning_rate']['milsestone'], 
                                                                      gamma=config['learning_rate']['gamma'])
        
        # define wandb log name
        self.name =  '_'.join([
            config['data']['pickle_name'].split('_')[0],
            # config['data']['pickle_name'].split('_')[2],
            str(self.lr),
            str(self.optim), 
            str(config['hidden_dim']), 
            # str(config['out_dim']), 
            str(config['Layer']),
            str(config['training']['batch_size']), 
            str(config['conv_type']),
            str(config['tag_kernal']),
            str(config['appnp_rate']),
            str(self.loss_name),
            str(config['data']['train_id'])
            
        ])
        
        
    def train(self):
        
        stop = False
        best = None
        
        if self.wandb:
            run=wandb.init(project=self.wandb_proj_name,entity='snu-mlvu-gnn',name=self.name)
            
        for epoch in range(self.epoch):
            self.model.train()
            print("Beginning training epch {}".format(epoch+1))
            
            loss_arr=[]
            for batch_idx, g_train in enumerate(self.train_data):
                
                if self.sampler :
                    
                    (input_nodes, seeds, blocks) = g_train
                    feature  = blocks[0].srcdata["feat"].to(self.device)
                    label = blocks[-1].dstdata['label'].to(self.device)
                    weight = blocks[-1].dstdata['pixel_num'].to(self.device)
                    # weight = g_train.ndata['pixel_num'].to(self.device)
                    blocks = [b.to(torch.device(self.device)) for b in blocks]
                    self.optimizer.zero_grad()
                    output = self.model(blocks, feature) 
                    
                elif self.multi_scale :
                
                    feature  = g_train.ndata['feat'].to(self.device)
                    label = g_train.ndata['label'].to(self.device)
                    weight = g_train.ndata['pixel_num'].to(self.device)
                    G = g_train.to(self.device)
                    self.optimizer.zero_grad()
                    # output = self.model(G, feature)
                    output_1, output_2, output_3 = self.model(G, feature)
                    # output_1, output_2, output_3 , output_4= self.model(G, feature)
                    
                else : 
                    feature  = g_train.ndata['feat'].to(self.device)
                    label = g_train.ndata['label'].to(self.device)
                    weight = g_train.ndata['pixel_num'].to(self.device)
                    G = g_train.to(self.device)
                    self.optimizer.zero_grad()
                    output = self.model(G, feature) 
                    
                label = F.one_hot(label.to(torch.int64), self.n_class) # one-hot encoding 추가
                labels = torch.max(label, 1)[1]
                if self.loss_name== 'ce':
                    loss = self.loss(output, labels)
                    
                elif self.loss_name =='splmse':
                    
                    pass
                
                elif self.loss_name =='splce':
                    if self.multi_scale :
                        loss_1 =self.loss(output_1, labels, weight, self.n_class)
                        loss_2 =self.loss(output_2, labels, weight, self.n_class)
                        loss_3 =self.loss(output_3, labels, weight, self.n_class)
                        loss = (loss_1 + loss_2  + loss_3)/3
                        
                    else : 
                        loss =self.loss(output, labels, weight, self.n_class)
                    
                loss.backward()
                loss_arr += [loss.item()]
                
                self.optimizer.step()
                if( batch_idx % self.eval_interval == 0 ):
                    print('    loss at batch {}: {}'.format(batch_idx, loss), flush=True)   
                    
            with torch.no_grad(): 
                self.model.eval()
                torch.cuda.empty_cache()
                val_loss = []
                val_acc = []
                print('Validation...')
                
                
                for batch_idx, g_valid in enumerate(self.valid_data):
                    if self.sampler :
                        
                        (input_nodes, seeds, blocks_val) = g_valid
                        feature_val  = blocks_val[0].srcdata['feat'].to(self.device)
                        label_val = blocks_val[-1].dstdata['label'].to(self.device)
                        weight_val = blocks_val[-1].dstdata['pixel_num'].to(self.device)
                        blocks_val = [b.to(torch.device(self.device)) for b in blocks_val]
                        self.optimizer.zero_grad()
                        output_val = self.model(blocks_val, feature_val) 
                    elif self.multi_scale :
                
                        feature_val  = g_valid.ndata['feat'].to(self.device)
                        label_val = g_valid.ndata['label'].to(self.device)
                        weight_val = g_valid.ndata['pixel_num'].to(self.device)
                        G_val = g_valid.to(self.device)
                        self.optimizer.zero_grad()
                        output_val_1, output_val_2, output_val_3 = self.model(G_val, feature_val)
                        
                    else : 
                        feature_val  = g_valid.ndata['feat'].to(self.device)
                        label_val = g_valid.ndata['label'].to(self.device)
                        weight_val = g_valid.ndata['pixel_num'].to(self.device)
                        G_val = g_valid.to(self.device)
                        self.optimizer.zero_grad()
                        output_val = self.model(G_val, feature_val) 
                    
                    
                    label_val_onehot = F.one_hot(label_val.to(torch.int64), self.n_class) # one-hot encoding 추가
                    labels_val = torch.max(label_val_onehot, 1)[1]
                    
                    if self.loss_name== 'ce':
                        loss_val = self.loss(output_val, labels_val)
                    elif self.loss_name =='splmse':
                        loss_val =self.loss(output_val, labels_val, label_val_onehot, weight_val, self.n_class)
                    elif self.loss_name =='splce':
                        if self.multi_scale:
                            loss_val_1 =self.loss(output_val_1, labels_val, weight_val, self.n_class)
                            loss_val_2 =self.loss(output_val_2, labels_val, weight_val, self.n_class)
                            loss_val_3 =self.loss(output_val_3, labels_val, weight_val, self.n_class)
                    
                            loss_val = (loss_val_1 + loss_val_2 + loss_val_3) /3
                    
                        else : 
                            loss_val =self.loss(output_val, labels_val, weight_val, self.n_class)
                        
                    
                    val_loss.append(loss_val.item())
                    
                    # calculate accuracy
                    if self.multi_scale :
                        output_val = (output_val_1 + output_val_2 + output_val_3)/3
                        pred = output_val.argmax(dim=1, keepdim=True)
                    else : 
                        pred = output_val.argmax(dim=1, keepdim=True)
                    crr = pred.eq(label_val.view_as(pred)).sum().item()
                    acc = crr / len(pred)
                    val_acc.append(acc)
                    
            print("validation loss : {:.6f}".format( np.mean(val_loss) ))
            print("validation acc : {:.6f}".format( np.mean(val_acc) ))
            if self.wandb:
                wandb.log({'train_loss':np.mean(loss_arr),
                           'val_loss':np.mean(val_loss),
                           'val_acc':np.mean(val_acc), 
                           "lr": get_lr(self.optimizer), 
                           'batch':self.batch_size, 
                           'hidden_dim':self.config['hidden_dim'],
                           'out_dim':self.config['out_dim'],
                           'layers':self.config['Layer'],
                          })

            
            
            PATH_ckpt = self.save_dir + "epoch_" + str(epoch+1).zfill(3) + '_ckpt.pth'
            patience = self.patience

            val_loss = np.mean(val_loss)
            
    
            if best is None :
                best = val_loss 
                counter = 0
            else : 
                if best < val_loss:
                    counter += 1
                else :
                    best = val_loss
                    if self.save_best_epoch_pth :
                        self.save_model(PATH_ckpt)
                    counter = 0
                    print("")
                    print("Model saved at epoch : {0:d}, validation loss : {1:.6f}".format(epoch+1,val_loss))
                    
            if counter >= patience :
                stop = True
            if stop:
                print("EarlyStopping Trigger")
                break
            if self.config['learning_rate']['scheduler']:
                self.scheduler.step()
            print("")
            print("")
        self.save_model(self.save_dir+'final.pth')

        print("training completed")
        if self.wandb :
            run.finish()
        return True
    
    
    def save_model(self,pth):
        """Save the final model output."""
        if isinstance(self.model, torch.nn.DataParallel):
            torch.save(self.model.module.state_dict(),pth)
        else:
            torch.save(self.model.state_dict(), pth)
def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']
def parse(path):
    with open(path, 'r') as f:
        config = yaml.safe_load(f)
        f.close()
    return config

def main():
    now = int(time.time())
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_yaml','-yml', type=str, default='train_uav.yaml')
    parser.add_argument('--tag_kernal','-tk', type=int)
    parser.add_argument('--conv_type','-conv', type=str)
    parser.add_argument('--hidden_dim','-dim', type=int)
    parser.add_argument('--Layer','-layer', type=int)
    parser.add_argument('--appnp_rate','-appnp', type=float)
    parser.add_argument('--gpu_id','-gpu', type=int)
    parser.add_argument('--batch_size','-bs', type=int)
    parser.add_argument('--batch_norm','-bn', type=str)
    


    args = parser.parse_args()
    config_path = f'yml/{args.config_yaml}'    
              
    config = parse(config_path)
    ## add more config
    config['data']['checkpoint_dir'] = config['data']['result_dir']+ f'/ckpt_{now}/'
    config['data']['train_id'] = str(now)
    config['tag_kernal'] = args.tag_kernal
    config['conv_type'] = args.conv_type
    config['hidden_dim'] = args.hidden_dim
    config['out_dim'] = args.hidden_dim
    config['Layer'] = args.Layer
    config['appnp_rate'] = args.appnp_rate
    config['batch_norm'] = args.batch_norm
    
    config['training']['batch_size'] = args.batch_size
    config['training']['gpu']['id'] = args.gpu_id
    
    
    
    if config['sampler']['sampler_true'] :
        sampler_list = config['sampler']['sampler_neighbor']
        print("Use Neighborhood sampler : {}".format(sampler_list))
        
        
        assert len(sampler_list) == int(config['Layer']), "The number of layers is not correlated!!!!"
        
    if config['wandb'] : 
        wandb.login()
    os.environ['CUDA_VISIBLE_DEVICES']=str(config['training']['gpu']['id'] )    
    trainer = Trainer(config=config)
    trainer.train()


if __name__ == "__main__":
    main()