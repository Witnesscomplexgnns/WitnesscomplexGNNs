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
parser.add_argument('--lm_perc',type=float,default=0.25,help='%nodes as landmarks')
parser.add_argument('--resolution',type = int, default = 50, help='resolution of PI')
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
		landmarks,dist_to_cover,cover = getLandmarksbynumL(G, L = L,heuristic=heuristic)
		local_pd = [None]*len(G.nodes)
		for u in cover:
			G_cv = nx.Graph()
			cv = set(cover[u])
			for v in cv:
				for w in G.neighbors(v):
					if w in cv:
						G_cv.add_edge(v,w)
			len_cv = len(cv)
			# print('|cover| (',u,') => ',len_cv)
			local_landmarks, local_dist_to_cover, _ = getLandmarksbynumL(G_cv, int(len_cv*args.lm_perc), heuristic = heuristic)
			DSparse,INF = get_sparse_matrix(G_cv,local_dist_to_cover,local_landmarks)
			resultsparse = ripser(DSparse, distance_matrix=True)
			resultsparse['dgms'][0][resultsparse['dgms'][0] == math.inf] = INF
			PD = resultsparse['dgms'][0] # H0
			PI = persistence_image(PD, resolution = [args.resolution, args.resolution]).reshape(1, args.resolution, args.resolution)
			# print('local_pd: ',len(local_pd),' ',u)
			# print(u,' has Nan: ',np.isnan(PI).any())
			if np.isnan(PI).any():
				PI = np.ones((1,args.resolution,args.resolution))*(10**-8) 
			local_pd[u] = PI 
			for v in cv:
				local_pd[v] = PI # copy topological features of landmarks to the witnesses
		for i,_ in enumerate(local_pd):
			if local_pd[i] is None:
				local_pd[i] = np.ones((1,args.resolution,args.resolution))*(10**-8) 
				# print('local_pd is none for node : ',i)
			# print(local_pd[i].shape)
		DSparse,INF = get_sparse_matrix(G,dist_to_cover,landmarks) # Construct sparse LxL matrix
		resultsparse = ripser(DSparse, distance_matrix=True)
		resultsparse['dgms'][0][resultsparse['dgms'][0] == math.inf] = INF # Replacing INFINITY death with a finite number.
		PD = resultsparse['dgms'][0] # H0
		
		with open(dataset_name+'.pd.pkl','wb') as f:
			pickle.dump(PD,f)
		# with open(dataset_name+'_local.pi.npz','wb') as f:
		# 	pickle.dump(local_pd,f)
		# PI = np.array(local_pd)
		PI = np.concatenate(local_pd)
		# print('local pI shape: ',PI.shape)
		np.savez_compressed(dataset_name + '_localPI.npz', PI)
	return PD

# load dataset
dataset_name = 'cora'
perturbed_adj = sp.load_npz('data/' + dataset_name + '/' + dataset_name + '_meta_adj_'+str(args.ptb_rate)+'.npz')
# adj, _, _ = load_npz('data/' + dataset_name + '/' + dataset_name + '.npz')
# G = nx.from_numpy_matrix(adj.toarray()[:100, :100]) # G is a sub-matrix of the input Adj matrix of CORA
G = nx.from_numpy_matrix(perturbed_adj.toarray())
dataset_name2 = 'data/' + dataset_name + '/' + dataset_name+"_"+str(args.ptb_rate)
PD = computeLWfeatures(G,dataset_name2, landmarkPerc=args.lm_perc, heuristic = 'degree')
# resolution_size = 50
PI = persistence_image(PD, resolution = [args.resolution, args.resolution])
PI = PI.reshape(1, args.resolution, args.resolution)
np.savez_compressed('data/' + dataset_name + '/' + dataset_name+"_"+str(args.ptb_rate) + '_PI.npz', PI)