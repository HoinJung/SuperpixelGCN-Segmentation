import pandas as pd 
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
import numpy as np


def class_weighted_loss(pred, label, superpixel_w, n_classes):

    V = label.size(0)
    label_count = torch.bincount(label)
    label_count = label_count[label_count.nonzero()].squeeze()
    cluster_sizes = torch.zeros(n_classes).long().to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    cluster_sizes[torch.unique(label)] = label_count

    weight = (V - cluster_sizes).float() / V
    weight *= (cluster_sizes>0).float()

    criterion = nn.CrossEntropyLoss(weight=weight)
    loss = criterion(pred, label)

    return loss    



def superpixel_penalty_loss(pred, label, superpixel_w, n_classes):


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

def tversky_loss_ori(inputs, targets):
    alpha = 0.5
    beta = 0.5 
    smooth=1
    #comment out if your model contains a sigmoid or equivalent activation layer
    inputs = F.softmax(inputs, dim = 1)  
    
    inputs = inputs.view(-1)
    targets = targets.view(-1)

    TP = (inputs * targets).sum()    
    FP = ((1-targets) * inputs).sum()
    FN = (targets * (1-inputs)).sum()

    Tversky = (TP + smooth) / (TP + alpha*FP + beta*FN + smooth)  
    Tversky = torch.log(Tversky.clamp_min(0.0001))
    return -Tversky  


def tversky_loss(inputs, targets, index):
    alpha = 0.5
    beta = 0.5 
    smooth=1
    #comment out if your model contains a sigmoid or equivalent activation layer
    inputs = F.softmax(inputs, dim = 1)  
    
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
    
def select_loss(loss):
    if loss == 'ce' :
        selected_loss =  nn.CrossEntropyLoss()
    elif loss == 'wce' : 
        selected_loss = class_weighted_loss()
    elif loss == 'spl' :
        selected_loss = superpixel_penalty_loss()
    elif loss == 'tvs' :
        selected_loss = tversky_loss()
    return selected_loss