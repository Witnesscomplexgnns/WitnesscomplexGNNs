import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import pandas as pd
import networkx as nx
import os,pickle
from torch_geometric.data import Data
import torch_geometric.datasets
import torch_geometric.transforms as T
import sys
from utils import load_data
from persistence_image import persistence_image
from torch_geometric.utils import to_networkx
from utils import load_npz
from lazywitness import * 
import argparse 
import scipy.sparse as sp

parser = argparse.ArgumentParser()
parser.add_argument('--ptb_rate', type=float, default=0.05,  help='pertubation rate')
args = parser.parse_args()

def computeLWfeatures(G,dataset_name, landmarkPerc=0.25,heuristic = 'degree'):
	"""
	 Computes LW persistence pairs / if pre-computed loads&returns them. 
	landmarkPerc = %nodes fo G to be picked for landmarks. [ Currently we are selecting landarmks by degree]
	heuristic = landmark selection heuristics ('degree'/'random') 
	Returns only H0 PD	
	"""
	if os.path.isfile(dataset_name+'.pd.pkl'):
		with open(dataset_name+'.pd.pkl','rb') as f:
			PD = pickle.load(f)
	else:	
		L = int(len(G.nodes)*landmarkPerc) # Take top 25% maximal degree nodes as landmarks
		landmarks,dist_to_cover = getLandmarksbynumL(G, L = L,heuristic='degree')
		DSparse,INF = get_sparse_matrix(G,dist_to_cover,landmarks) # Construct sparse LxL matrix
		resultsparse = ripser(DSparse, distance_matrix=True)
		resultsparse['dgms'][0][resultsparse['dgms'][0] == math.inf] = INF # Replacing INFINITY death with a finite number.
		PD = resultsparse['dgms'][0] # H0
		
		with open(dataset_name+'.pd.pkl','wb') as f:
			pickle.dump(PD,f)
	return PD

# load dataset
dataset_name = 'cora'
perturbed_adj = sp.load_npz('data/' + dataset_name + '/' + dataset_name + '_meta_adj_'+str(args.ptb_rate)+'.npz')
# adj, _, _ = load_npz('data/' + dataset_name + '/' + dataset_name + '.npz')
# G = nx.from_numpy_matrix(adj.toarray()[:100, :100]) # G is a sub-matrix of the input Adj matrix of CORA
G = nx.from_numpy_matrix(perturbed_adj.toarray()[:100, :100])
PD = computeLWfeatures(G, 'data/' + dataset_name + '/' + dataset_name+"_"+str(args.ptb_rate), landmarkPerc=0.25, heuristic = 'random')
resolution_size = 100
PI = persistence_image(PD, resolution = [resolution_size, resolution_size])
PI = PI.reshape(1, resolution_size, resolution_size)
np.savez_compressed('data/' + dataset_name + '/' + dataset_name+"_"+str(args.ptb_rate) + '_PI.npz', PI)
