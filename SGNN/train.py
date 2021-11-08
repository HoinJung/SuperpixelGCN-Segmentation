import pandas as pd 
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import dgl
from GraphSage import GraphSageNet, GraphSageNet_sampler
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
from loss import select_loss

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
       
        self.eval_interval = config['training']['eval_interval']
        
            
        # code for wandb    
        self.wandb = config['wandb']
        self.wandb_proj_name = config['wandb_proj_name']
        
        # make models and data 
        self.train_data , self.valid_data = data_generator(config)
        self.loss = select_loss(config['loss'])
        
        if self.sampler :
            # sampler mode
            self.model = GraphSageNet_sampler(config).to(self.device)
        else : 
            # graph(?) mode
            self.model = GraphSageNet(config).to(self.device)
        # if you want to use multi-GPU
        # self.model = torch.nn.DataParallel(self.model)       
        
        if self.optim == 'Adam' : 
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        elif self.optim == 'SGD' : 
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr)
            
        # define wandb log name
        self.name =  '_'.join([
            config['data']['pickle_name'].split('_')[0],
            config['data']['pickle_name'].split('_')[2],
            str(self.lr),
            str(self.optim), 
            str(config['hidden_dim']), 
            str(config['out_dim']), 
            str(config['Layer']),
            str(config['training']['batch_size']), 
            'class_weighted',
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
                    blocks = [b.to(torch.device(self.device)) for b in blocks]
                    self.optimizer.zero_grad()
                    output = self.model(blocks, feature) 
                else : 
                    feature  = g_train.ndata['feat'].to(self.device)
                    label = g_train.ndata['label'].to(self.device)
                    weight = g_train.ndata['pixel_num'].to(self.device)
                    G = g_train.to(self.device)
                    self.optimizer.zero_grad()
                    output = self.model(G, feature) 
                label = F.one_hot(label.to(torch.int64), self.n_class) # one-hot encoding 추가
                label = torch.max(label, 1)[1]
                loss = self.loss(output, label)
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
                        blocks_val = [b.to(torch.device(self.device)) for b in blocks_val]
                        self.optimizer.zero_grad()
                        output_val = self.model(blocks_val, feature_val) 
                    else : 
                        feature_val  = g_valid.ndata['feat'].to(self.device)
                        label_val = g_valid.ndata['label'].to(self.device)
                        G_val = g_valid.to(self.device)
                        self.optimizer.zero_grad()
                        output_val = self.model(G_val, feature_val) 
                    label_val_onehot = F.one_hot(label_val.to(torch.int64), self.n_class) # one-hot encoding 추가
                    label_val = torch.max(label_val_onehot, 1)[1]
                    loss_val = self.loss(output_val, label_val)
                    val_loss.append(loss_val.item())
                    
                    # calculate accuracy
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
                           'layers':self.config['L'],
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
                    self.save_model(PATH_ckpt)
                    counter = 0
                    print("")
                    print("Model saved at epoch : {0:d}, validation loss : {1:.6f}".format(epoch+1,val_loss))
                    
            if counter >= patience :
                stop = True
            if stop:
                print("EarlyStopping Trigger")
                break

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
    parser.add_argument('--config_yaml','-yml', type=str, default='train_hi.yaml')
    args = parser.parse_args()
    config_path = f'yml/{args.config_yaml}'
    
              
    config = parse(config_path)
    ## add more config
    config['data']['checkpoint_dir'] = config['data']['result_dir']+ f'/ckpt_{now}/'
    config['data']['train_id'] = str(now)
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