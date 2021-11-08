#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import pandas as pd
import glob
import numpy as np
from skimage.segmentation import slic, mark_boundaries
from tqdm import tqdm
import networkx as nx
from UAVidToolKit.colorTransformer import UAVidColorTransformer
import os
import sys
# sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname('./superpixels/superpixel_fcn'))))
sys.path.append(os.path.abspath('./superpixels/superpixel_fcn'))
# sys.path.append(os.path.abspath(os.path.join(__file__,'./superpixels/superpixel_fcn')))
# print(sys.path)
# lib_path = os.path.abspath(os.path.join(__file__, '..', '..', '..', 'lib'))
import torch.backends.cudnn as cudnn
import models
# import models
import torchvision.transforms as transforms
import flow_transforms
from skimage.io import imread, imsave
from loss import *
import time
import random


import matplotlib.pyplot as plt



# phases = ['train']
desired_nodes = 10000
NUM_FEATURES =5
NUM_CLASSES =1
NP_TORCH_FLOAT_DTYPE = np.float32
NP_TORCH_LONG_DTYPE = np.int64
downsize = 16
@torch.no_grad()
def test(img, model):
      # Data loading code
    
    input_transform = transforms.Compose([
        flow_transforms.ArrayToTensor(),
        transforms.Normalize(mean=[0,0,0], std=[255,255,255]),
        transforms.Normalize(mean=[0.411,0.432,0.45], std=[1,1,1]),
        # transforms.CenterCrop((576,560))
    ])

    img_ = img
    H, W, _ = img_.shape
    H_, W_  = int(np.ceil(H/16.)*16), int(np.ceil(W/16.)*16)

    # get spixel id
    n_spixl_h = int(np.floor(H_ / downsize))
    n_spixl_w = int(np.floor(W_ / downsize))

    spix_values = np.int32(np.arange(0, n_spixl_w * n_spixl_h).reshape((n_spixl_h, n_spixl_w)))
    spix_idx_tensor_ = shift9pos(spix_values)

    spix_idx_tensor = np.repeat(
      np.repeat(spix_idx_tensor_, downsize, axis=1), downsize, axis=2)

    spixeIds = torch.from_numpy(np.tile(spix_idx_tensor, (1, 1, 1, 1))).type(torch.float).cuda()

    n_spixel =  int(n_spixl_h * n_spixl_w)


    img = cv2.resize(img_, (W_, H_), interpolation=cv2.INTER_CUBIC)
    img1 = input_transform(img)
    ori_img = input_transform(img_)

    # # compute output
    # tic = time.time()
    output = model(img1.cuda().unsqueeze(0))
    # toc = time.time() - tic

    # assign the spixel map
    curr_spixl_map = update_spixl_map(spixeIds, output)
    ori_sz_spixel_map = F.interpolate(curr_spixl_map.type(torch.float), size=( H_,W_), mode='nearest').type(torch.int)

    mean_values = torch.tensor([0.411, 0.432, 0.45], dtype=img1.cuda().unsqueeze(0).dtype).view(3, 1, 1)
    spixel_viz, spixel_label_map = get_spixel_image((ori_img + mean_values).clamp(0, 1), ori_sz_spixel_map.squeeze(), n_spixels= n_spixel,  b_enforce_connect=True)


    # print('sfcn....end')

    return spixel_label_map


def get_graph_from_segments(image, gt, segments):
    # load the image and convert it to a floating point data type
    # print('graph....start')
    num_nodes = np.max(segments)
    nodes = {
        node: {
            "rgb_list": [],
            "pos_list": [],
            "label_list": [],
        } for node in range(num_nodes+1)
    }
    
    height = image.shape[0]
    width = image.shape[1]
    for y in range(height):
        for x in range(width):
            
            node = segments[y,x]
            
            rgb = image[y,x,:]
            
            label = gt[y,x]
            pos = np.array([float(x)/width,float(y)/height])
            nodes[node]["rgb_list"].append(rgb)
            nodes[node]["pos_list"].append(pos)
            nodes[node]["label_list"].append(label)

        #end for
    #end for

    G = nx.Graph()
    pos_list = []
    rgb_list = []
    label_list = []
    for node in nodes:
        
        
        # if len(new_nodes[node]["rgb_list"])<=1:
        #     print(len(new_nodes[node]["rgb_list"]))
        nodes[node]["rgb_list"] = np.stack(nodes[node]["rgb_list"])
        nodes[node]["pos_list"] = np.stack(nodes[node]["pos_list"])
        nodes[node]["label_list"] = np.stack(nodes[node]["label_list"])
        
        # rgb
        rgb_mean = np.mean(nodes[node]["rgb_list"], axis=0)
        
        # Pos
        pos_mean = np.mean(nodes[node]["pos_list"], axis=0)
        
        # label
        uniq, cnts = np.unique(nodes[node]["label_list"],return_counts=True)

        label_max = uniq[cnts.argmax()]

        
        rgb_list.append(rgb_mean)
        pos_list.append(pos_mean)
        label_list.append(label_max)
        
            
        features = np.concatenate(
          [
            np.reshape(rgb_mean, -1),
            
            np.reshape(pos_mean, -1),
            
          ]
        )

        G.add_node(node, features = list(features))
     #end
    
    # From https://stackoverflow.com/questions/26237580/skimage-slic-getting-neighbouring-segments
    segments_ids = np.unique(segments)

    # centers
    # centers = np.array([np.mean(np.nonzero(segments==i),axis=1) for i in segments_ids])
    # centers
    segments_ids = np.unique(segments)
    segments_ids = torch.from_numpy(segments_ids)
    segments_ids = segments_ids.to(torch.device(device))
    
    segments_torch = torch.from_numpy(segments)
    segments_torch = segments_torch.to(torch.device(device))
    centers = np.array((torch.mean(torch.nonzero(segments_torch==i),axis=1) for i in segments_ids))
    # centers = np.array((np.mean(np.nonzero(segments==i),axis=1) for i in segments_ids))
    # for i in segments_ids:
    #     mean_val = np.nonzero(segments==i)
    #     centers = np.array([np.mean(mean_val,axis=1)])
        
        
    vs_right = np.vstack([segments[:,:-1].ravel(), segments[:,1:].ravel()])
    vs_below = np.vstack([segments[:-1,:].ravel(), segments[1:,:].ravel()])
    bneighbors = np.unique(np.hstack([vs_right, vs_below]), axis=1)

    # Adjacency loops
    for i in range(bneighbors.shape[1]):
        if bneighbors[0,i] != bneighbors[1,i]:
            G.add_edge(bneighbors[0,i],bneighbors[1,i])

    # Self loops
    for node in nodes:
        G.add_edge(node,node)
    
    n = len(G.nodes)
    m = len(G.edges)
    h = np.zeros([n,NUM_FEATURES]).astype(NP_TORCH_FLOAT_DTYPE)
    edges = np.zeros([2*m,2]).astype(NP_TORCH_LONG_DTYPE)
    for e,(s,t) in enumerate(G.edges):
        edges[e,0] = s
        edges[e,1] = t
        
        edges[m+e,0] = t
        edges[m+e,1] = s
    #end for
    for i in G.nodes:

        h[i,:] = G.nodes[i]["features"]
    #end for
    # print('graph....end')
    return G, h, edges, rgb_list, pos_list, label_list



data_list= []
phase = 'test'
path = f'../cityscape/leftImg8bit/{phase}'

file_list = glob.glob(f'{path}/**/*.png',recursive=True)
pretrained = './superpixels/superpixel_fcn/pretrain_ckpt/SpixelNet_bsd_ckpt.tar'

os.environ['CUDA_VISIBLE_DEVICES'] = '2'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
 # create model
network_data = torch.load(pretrained)
print("=> using pre-trained model '{}'".format(network_data['arch']))
model = models.__dict__[network_data['arch']]( data = network_data).cuda()
model.eval()
# args.arch = network_data['arch']
cudnn.benchmark = True

for file in tqdm(file_list) : 
    filedir = file.split('/')
    type_img = filedir[-4]
    img_name = filedir[-1].split('.')[0]
    img_id = "_".join([type_img,  img_name])
    if type_img == 'leftImg8bit':
        file_path = file
        # print(file_path)
        gt_path = file.replace('leftImg8bit','gtFine').replace('.png','_labelIds.png')
        img = imread(file_path)
        gt = imread(gt_path)
        
        segments = test(img, model)
        
        asegments = np.array(segments)
        
        # 이부분이 안맞을 가능성이 있음.. 현재 G가 뭔가 사이즈가 안맞음
        # clrEnc = UAVidColorTransformer()
        # trainId = clrEnc.transform(gt, dtype=np.uint8)
        
        G, h, edges, rgb_list, pos_list, label_list_gt = get_graph_from_segments(img, gt, asegments)            

        data_list.append([file_path, img_id, asegments,gt_path,G, h, edges, rgb_list, pos_list,label_list_gt])
        # print('{} %'.format(cnt * 100 * 2/ len(file_list)))


# In[ ]:


# segments_id = np.unique(asegments)
# segments_id
# for i in segments_id:
    
# centers = np.array([np.mean(np.nonzero(segments==i),axis=1) for i in segments_id])
# # for i in segments_id:
# #     print(i)


# In[ ]:



df = pd.DataFrame(data_list)
df.columns=['image_path','id','superpixel_segment','gt_path','G','feature','edges','rgb_mean','pos_mean','label_gt']
# df['label']=df['id'].str.replace('Images','Labels')
# df['label_path']=df['image_path'].str.replace('Images','Labels')
# df.to_csv('train.csv',index=False)
df.to_pickle(f'city_sfcn_{phase}.pickle')


# In[ ]:


# import glob
# data_list= []
# phase = 'train'
# path = f'../cityscape/leftImg8bit/{phase}'

# file_list = glob.glob(f'{path}/**/*.png',recursive=True)
# print(len(file_list))
# # pretrained = './superpixels/superpixel_fcn/pretrain_ckpt/SpixelNet_bsd_ckpt.tar'
# # cityscape\leftImg8bit_trainvaltest\leftImg8bit\train\bochum\bochum_000000_000313_gtFine_labelIds.png
# # cityscape\gtFine_trainvaltest\gtFine\train\bochum\bochum_000000_000313_leftImg8bit.png


# In[ ]:


# from tqdm import tqdm
# for file in tqdm(file_list) : 
#     filedir = file.split('/')
#     type_img = filedir[-4]
#     img_name = filedir[-1].split('.')[0]
#     img_id = "_".join([type_img,  img_name])
#     if type_img == 'leftImg8bit':
#         file_path = file
#         # print(file_path)
#         gt_path = file.replace('leftImg8bit','gtFine').replace('.png','_labelIds.png')
#         # img = imread(file_path)
#         # gt = imread(gt_path)


# In[ ]:


# gt = imread(gt_path) 
# from skimage.io import imshow
# print(gt.shape)
# imshow(gt)


# In[ ]:




