import pandas as pd 
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import dgl
from GraphSage import GraphSageNet
import easydict 
import time
import os
import wandb
import time
from tqdm import tqdm
class Tester(object):
    def __init__(self, config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = config['training']['batch_size']
        self.train_id =  config['train_id']
        self.model_path = config['data']['checkpoint_dir'] + 'ckpt_'+str(self.train_id)+'/'+config['data']['ckpt_file']
        self.save_dir = config['data']['checkpoint_dir']; os.makedirs(self.save_dir, exist_ok = True)
        self.n_class = config['training']['n_classes']
        self.sampler = config['sampler']['sampler_true']
        # self.test_data= data_test_generator(config)
        self.result_save_dir = 'eval_result_weight/'; os.makedirs(self.result_save_dir, exist_ok = True)
        self.model = GraphSageNet(config).to(self.device)
        # self.model = torch.nn.DataParallel(self.model)
        self.result_name = 'result'+'_'+str(self.train_id)
        
        
    def test(self):
        self.model.load_state_dict(torch.load(self.model_path,map_location=self.device))
        self.model.eval()
        torch.cuda.empty_cache()
        self.raw_data = pd.read_pickle(self.config['data']['pickle_dir']+ self.config['data']['pickle_name'])  
        with torch.no_grad(): 
            
            
            print('Test...')
            test_acc = []
            save_test_result = []
            for idx, row in tqdm(self.raw_data.iterrows()):
                spixel= row['superpixel_segment']
                gt_path = row['gt_path']
                G = row['G']
                feature = row['feature']
                feature[:,:3] = feature[:,:3] / 255 # rgb normalization
                edges = row['edges']
                label_gt = row['label_gt']
                num_nodes = len(label_gt)
                edges_src = torch.tensor(edges[:,0])
                edges_dst = torch.tensor(edges[:,1])
                dgel_graph = dgl.graph((edges_src , edges_dst), num_nodes=num_nodes, idtype=torch.int32)
                dgel_graph.ndata['feat'] = torch.from_numpy(feature)
                dgel_graph.ndata['label'] = torch.from_numpy(np.array(label_gt))
                dgel_graph = dgl.add_self_loop(dgel_graph)
                
                
                feature_test  = dgel_graph.ndata['feat'].to(self.device)
                label_test = dgel_graph.ndata['label'].to(self.device)
                G_test = dgel_graph.to(self.device)
                output_test = self.model(G_test, feature_test) 
                label_test_onehot = F.one_hot(label_test.to(torch.int64), self.n_class) 
                label_test = torch.max(label_test_onehot, 1)[1]
                
                # calculate accuracy
                pred = output_test.argmax(dim=1, keepdim=True)
                crr = pred.eq(label_test.view_as(pred)).sum().item()
                acc = crr / len(pred)
                test_acc.append(acc)
                feature_test = feature_test.cpu().numpy()
                label_test = label_test.cpu().numpy()
                
                output_test = output_test.cpu().numpy()
                # label_test = label_test.cpu().numpy()
                pred = pred.cpu().numpy()
                
                save_test_result.append([spixel, gt_path, feature_test,label_test,G,output_test,label_gt,pred])
            
            print("validation acc : {:.6f}".format( np.mean(test_acc) ))
            df = pd.DataFrame(save_test_result)
            df.columns = ['spixel','gt_path','feature_test','label_test','G','output_test','label_gt','pred']
            print('Save Dataframe...')
            df.to_pickle(self.result_save_dir + self.result_name + '.pickle')
            print('Done!')
        
    
    

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
    os.environ['CUDA_VISIBLE_DEVICES']=str(config['training']['gpu']['id'] )    
    


    wandb_name = config['test']['wandb_name']
    wandb_names = wandb_name.split('_')
    config['hidden_dim'] = int(wandb_names[4])
    config['out_dim'] = int(wandb_names[5])
    config['Layer'] = int(wandb_names[6])
    config['train_id'] = wandb_names[-1]
    print(f"train id : {config['train_id']}, hidden dim : {config['hidden_dim']}, out_dim : {config['out_dim']} , Layers : {config['Layer']}")
    tester = Tester(config=config)
    tester.test()


if __name__ == "__main__":
    main()