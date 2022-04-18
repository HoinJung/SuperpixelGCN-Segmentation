import os
import pandas as pd
import glob
import numpy as np
from skimage.segmentation import slic, mark_boundaries
from tqdm import tqdm
import argparse
import networkx as nx
from UAVidToolKit.colorTransformer import UAVidColorTransformer
import os
from dgl.data.utils import save_graphs
from dgl.data.utils import load_graphs
import dgl
import os
import sys

sys.path.append(os.path.abspath('./superpixels/superpixel_fcn'))
import torch.backends.cudnn as cudnn
import models
import torchvision.transforms as transforms
import flow_transforms
from skimage.io import imread, imsave
from loss import *
import time
import random
import matplotlib.pyplot as plt




parser = argparse.ArgumentParser(description='Argparse Tutorial')
parser.add_argument('--num_nodes','-num', type=int, default=10000)
parser.add_argument('--num_features','-feat', type=int, default=5)
parser.add_argument('--num_classes','-class', type=int, default=8)
args = parser.parse_args()


desired_nodes = args.num_nodes
NUM_FEATURES = args.num_features
NUM_CLASSES = args.num_classes
NP_TORCH_FLOAT_DTYPE = np.float32
NP_TORCH_LONG_DTYPE = np.int64
downsize = 16
@torch.no_grad()
def test(img, model):
      # Data loading code
    
    input_transform = transforms.Compose([
        flow_transforms.ArrayToTensor(),
        transforms.Normalize(mean=[0,0,0], std=[255,255,255]),
        transforms.Normalize(mean=[0.411,0.432,0.45], std=[1,1,1])
    ])

#     img_file = img_paths[idx]
#     load_path = img_file
#     imgId = os.path.basename(img_file)[:-4]

#     # may get 4 channel (alpha channel) for some format
#     img_ = imread(load_path)[:, :, :3]
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


    return spixel_label_map


def get_graph_from_segments(image, gt, segments):
    # load the image and convert it to a floating point data type
    # print('graph....start')
    num_nodes = np.max(segments)
    idx,cnt = np.unique(segments,return_counts=True)
    
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
        nodes[node]["rgb_list"] = np.stack(nodes[node]["rgb_list"])
        
        nodes[node]["pos_list"] = np.stack(nodes[node]["pos_list"])
        nodes[node]["label_list"] = np.stack(nodes[node]["label_list"])
        
        # rgb
        rgb_mean = np.mean(nodes[node]["rgb_list"], axis=0)
        
        # Pos
        pos_mean = np.mean(nodes[node]["pos_list"], axis=0)
        
        # label
        uniq, cnts = np.unique(nodes[node]["label_list"],return_counts=True)
        
        # print(cnts.argmax())
        
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
    segments_ids = torch.from_numpy(segments_ids)
    segments_ids = segments_ids.to(torch.device(device))
    
    segments_torch = torch.from_numpy(segments)
    segments_torch = segments_torch.to(torch.device(device))
    
    # centers
    centers = np.array((torch.mean(torch.nonzero(segments_torch==i),axis=1) for i in segments_ids))

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

    return G, h, edges, rgb_list, pos_list, label_list



phases = ['train','val']
os.environ['CUDA_VISIBLE_DEVICES']='2'
for phase in phases :
    data_list= []    
    path = f'../data/uavid/uavid_{phase}/'
    file_list = glob.glob(f'{path}/**/Images/*.png',recursive=True)
    pretrained = './superpixels/superpixel_fcn/pretrain_ckpt/SpixelNet_bsd_ckpt.tar'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
     # create model
    network_data = torch.load(pretrained)
    print("=> using pre-trained model '{}'".format(network_data['arch']))

    print(f"Phase : {phase}")
    model = models.__dict__[network_data['arch']]( data = network_data).cuda()
    model.eval()
    args.arch = network_data['arch']
    cudnn.benchmark = True
    cnt = 0  
    for file in tqdm(file_list) :
        img_path = file
        gt_path = img_path.replace('Images','Labels')
        img = imread(file)
        gt = imread(gt_path)

        segments = test(img, model)
        clrEnc = UAVidColorTransformer()
        gt_label = clrEnc.transform(gt, dtype=np.uint8)
        asegments = np.array(segments)
        cnt +=1
        G, h, edges, rgb_list, pos_list, label_list_gt = get_graph_from_segments(img, gt_label, asegments)            
        data_list.append([img_path, asegments,G, h, edges, label_list_gt])

    df = pd.DataFrame(data_list)
    print(df.columns)
    df.columns=['id','superpixel_segment','G','feature','edges','label_gt']

    s_num_list = []
    for idx, row in df.iterrows():
        s_num = row['superpixel_segment']
        _, count = np.unique(s_num, return_counts=True)
        s_num_list.append(count)
    df['superpixel_num'] = np.array(s_num_list)
    if phase == 'train':
        df = df.drop(['superpixel_segment'],axis=1)
    if phase == 'val':
        df.to_pickle(f'../pickles/uav_{phase}_sfcn.pickle')
    graphs = []
    for idx, row in df.iterrows():

        G = row['G']
        feature = row['feature']
        feature[:,:3] = feature[:,:3] / 255 # rgb normalization
        count = row['superpixel_num']  
        edges = row['edges']
        label_gt = row['label_gt']
        num_nodes = len(label_gt)
        edges_src = torch.tensor(edges[:,0])
        edges_dst = torch.tensor(edges[:,1])
        dgel_graph = dgl.graph((edges_src , edges_dst), num_nodes=num_nodes, idtype=torch.int32)
        dgel_graph.ndata['feat'] = torch.from_numpy(feature)
        dgel_graph.ndata['pixel_num'] = torch.from_numpy(count)
        dgel_graph.ndata['label'] = torch.from_numpy(np.array(label_gt))
        # dgel_graph = dgl.remove_self_loop(dgel_graph)
        # dgel_graph = dgl.add_self_loop(dgel_graph)
        graphs.append(dgel_graph)
    graph_len = len(df)
    graph_labels = {"glabel": torch.tensor([i for i in range(graph_len)])}
    save_graphs(f'../pickles/uav_{phase}.bin', graphs, graph_labels)
        
