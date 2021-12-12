import pandas as pd 
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
import numpy as np

class CustomLoss():
    def __init__(self,loss_name, pred=None,label=None,labels=None,superpixel_w=None,n_classes=None):
        self.loss_name = loss_name
        # self.pred = pred
        # self.label = label
        # self.labels = labels
        # self.superpixel_w = superpixel_w   
        # self.n_classes = n_classes
    def select_loss(self):
        if self.loss_name == 'ce' :
            return nn.CrossEntropyLoss()
        elif self.loss_name == 'wce' : 
            return class_weighted_loss()
        elif self.loss_name == 'splce' :
            return superpixel_penalty_loss()
        elif self.loss_name == 'tvs' :
            return Tversky_loss()
        elif self.loss_name == 'splmse':
            return weighted_loss_mse()
        elif self.loss_name == 'splceclass':
            return superpixel_penalty_loss()
        
class weighted_loss_mse(nn.Module):
    def __init__(self):
        super().__init__()

        pass

    def forward(self, pred, label, labels, superpixel_w, n_classes):
        V =  label.size(0)
        label_count = torch.bincount(label)
        label_count = label_count[label_count.nonzero()].squeeze()

        cluster_sizes = torch.zeros(n_classes).long().to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        cluster_sizes[torch.unique(label)] = label_count

        weight = (V - cluster_sizes).float() / V
        weight *= (cluster_sizes>0).float()

        loss = weighted_mse_loss(pred, labels, weight)

        loss = torch.mean(loss, dim=1)


        epsilon = 0.0001
        superpixel_w = ((1+epsilon)/(torch.log(1/superpixel_w)+epsilon))
        w_loss = torch.multiply(superpixel_w, loss)
        w_loss = torch.mean(loss)

        return 10*w_loss  

def weighted_mse_loss(inputs, target, weight):
    return weight * (inputs - target) ** 2    

        
class superpixel_penalty_loss(nn.Module):
    def __init__(self):
        super().__init__()

        pass
    def forward(self, pred, label, superpixel_w, n_classes):
        # print(pred[0])
        V = label.size(0)
        label_count = torch.bincount(label)
        label_count = label_count[label_count.nonzero()].squeeze()
        cluster_sizes = torch.zeros(n_classes).long().to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        cluster_sizes[torch.unique(label)] = label_count

        weight = (V - cluster_sizes).float() / V
        weight *= (cluster_sizes>0).float()

        criterion = nn.CrossEntropyLoss(weight=weight, reduce=None, reduction='none')
        loss = criterion(pred, label)


        epsilon = 0.0001
        superpixel_w = ((1+epsilon)/(torch.log(1/superpixel_w)+epsilon))
        w_loss = torch.multiply(superpixel_w, loss)
        w_loss = torch.mean(loss)


        return w_loss    
    
    
    
class Tversky_loss(nn.Module):
    def __init__(self):
        super().__init__()

        pass
    def forward(self, pred, label, superpixel_w, n_classes):
        
        cal = 'train'
        Tversky = 0
        for indexs in range(n_classes):
            Tversky += tversky_loss(pred, label, index = indexs)

            if cal == 'val':   
                print('class:', indexs,'loss:', tversky_loss(pred, label, index = indexs))
        return 1 - Tversky/n_classes  
        
def tversky_loss(targets, inputs, index):
    alpha = 0.5
    beta = 0.5 
    smooth=1
    #comment out if your model contains a sigmoid or equivalent activation layer
    #print(inputs.size(), targets.size())
    targets = F.softmax(targets)  
    #print(inputs.size(), targets.size())
    
    inputs = inputs[:,index]
    targets = targets[:,index]

    #inputs = F.sigmoid(inputs)
    #print(inputs[0])
    #print(targets[0])
    #flatten label and prediction tensors
    #print(inputs.shape, targets.shape)

    inputs = inputs.view(-1)
    targets = targets.view(-1)
    #print(inputs.shape, targets.shape)

    #True Positives, False Positives & False Negatives
    TP = (inputs * targets).sum()    
    FP = ((1-targets) * inputs).sum()
    FN = (targets * (1-inputs)).sum()

    Tversky = (TP + smooth) / (TP + alpha*FP + beta*FN + smooth)  

    return Tversky      