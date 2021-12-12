import os
import pandas as pd
import glob
import numpy as np
from skimage.segmentation import slic, mark_boundaries
from skimage.io import imread
from tqdm import tqdm
import argparse
import networkx as nx
from UAVidToolKit.colorTransformer import UAVidColorTransformer
import torch
os.environ['CUDA_VISIBLE_DEVICES']='2'
parser = argparse.ArgumentParser(description='Argparse Tutorial')
parser.add_argument('--num_nodes','-num', type=int, default=17003)
parser.add_argument('--num_features','-feat', type=int, default=5)
parser.add_argument('--num_classes','-class', type=int, default=8)
args = parser.parse_args()

# phases = ['train']
desired_nodes = args.num_nodes
NUM_FEATURES = args.num_features
NUM_CLASSES = args.num_classes
NP_TORCH_FLOAT_DTYPE = np.float32
NP_TORCH_LONG_DTYPE = np.int64

def get_graph_from_segments(image, gt, segments):
    # load the image and convert it to a floating point data type
    os.environ['CUDA_VISIBLE_DEVICES'] = '2'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
    segments_ids = torch.from_numpy(segments_ids)
    segments_ids = segments_ids.to(torch.device(device))
    
    segments_torch = torch.from_numpy(segments)
    segments_torch = segments_torch.to(torch.device(device))
    centers = np.array((torch.mean(torch.nonzero(segments_torch==i),axis=1) for i in segments_ids))

    # centers
    # centers = np.array([np.mean(np.nonzero(segments==i),axis=1) for i in segments_ids])

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


def main():
    data_list= []
    phase = 'train'
    # for phase in phases :
    # path = f'../data/uavid/uavid_{phase}/'
    path = f'../data/uavid/uavid_train_patch(2048)/'
    file_list = glob.glob(f'{path}/**/*.png',recursive=True)
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cnt = 0
    print(file_list)
    for file in tqdm(file_list) : 
        filedir = file.split('/')
        # print(filedir)
        seq = filedir[3]
        type_img=filedir[-2]
        img_name = filedir[-1].split('.')[0]
        img_id = "_".join([type_img, seq, img_name])
        if type_img == 'Images':
            file_path = file
            gt_path = file.replace('Images','Labels')
            img = imread(file_path)
            gt = imread(gt_path)
            segments = slic(img,n_segments = desired_nodes, slic_zero = True)
            asegments = np.array(segments)
            # clrEnc = UAVidColorTransformer()
            # trainId = clrEnc.transform(gt, dtype=np.uint8)
            
            G, h, edges, rgb_list, pos_list, label_list_gt = get_graph_from_segments(img, gt, asegments)            
            cnt+=1
            data_list.append([file_path, img_id, asegments,gt_path,G, h, edges, rgb_list, pos_list,label_list_gt])
            print('{} %'.format(cnt * 100 * 2/ len(file_list)))
            # if cnt==4 :
            #     break
                
    df = pd.DataFrame(data_list)
    df.columns=['image_path','id','superpixel_segment','gt_path','G','feature','edges','rgb_mean','pos_mean','label_gt']
    # df['label']=df['id'].str.replace('Images','Labels')
    # df['label_path']=df['image_path'].str.replace('Images','Labels')
    # df.to_csv('train.csv',index=False)
    df.to_pickle(f'uav_train_slic_{desired_nodes}_{NUM_FEATURES}_{phase}_2048.pickle')



if __name__ == "__main__":
    main()