from ripser import ripser
from persim import plot_diagrams
import numpy as np
import matplotlib.pyplot as plt
from scipy import sparse
import time
import networkx as nx
import random 
import math

def get_distancetocover(G, nodes,_L,debug = False):
    dist_to_cover = {}
    for u in _L:
        dist_to_cover[u] = 0
    for u in nodes:
        # if u=='3':
        #     debug = True
        # else:
        #     debug = False
        if u in dist_to_cover:
            if (debug):
                print(u,' previously added to dist_to_cover')
            continue
        Queue = [u]
        dist = {u: 0}
        breakflag = False
        while len(Queue):
            v = Queue.pop(0)
            if (debug):
                print('pop ',v)
            for nbr_v in G.neighbors(v):
                dist_nv = dist[v] + 1
                if (debug):
                    print('nbr: ',nbr_v,' dist_nv: ',dist_nv)
                if dist.get(nbr_v, math.inf) > dist_nv:
                    if (debug):
                        print('condition check: ',dist.get(nbr_v, math.inf))
                    dist[nbr_v] = dist_nv
                    if nbr_v in _L:
                        if (debug):
                            print('nbr_v ',nbr_v,' in _L')
                        dist_to_cover[u] = dist[nbr_v]
                        breakflag = True
                        break
                    else:
                        Queue.append(nbr_v)
            if breakflag:
                break
        if breakflag is False: # distance to nearest neighbor in L is infinity, because disconnected.
            dist_to_cover[u] = math.inf
    return dist_to_cover

def getLandmarksbynumL(G, L = 2, heuristic = 'degree'):
    """ dist_nearest_nbr_inL[u] is the distance from u to its nearest nbr in L"""
    if heuristic == 'degree':
        _degreenodes = sorted([(G.degree[u],u) for u in G.nodes],reverse = True)
        _L = set([pair[1] for pair in _degreenodes[:L]])
        del _degreenodes
        dist_to_cover = get_distancetocover(G, G.nodes,_L)
        return _L, dist_to_cover
        
    if heuristic == 'random':
        _L = random.sample(G.nodes, k=L)
        dist_to_cover = get_distancetocover(G, G.nodes,_L)
        return _L, dist_to_cover
    
        
def getLandmarksbyeps(G, epsilon = 2, heuristic = 'epsmaxmin'):
    dist_to_cover = {}
    _L = []
    if heuristic=='epsmaxmin':
        marked = {}
        for u in G.nodes:
            dist_to_cover[u] = math.inf
            marked[u] = False
        num_marked = 0
        _N = len(G.nodes)
        while num_marked < _N:
            if num_marked == 0:
                u = list(G.nodes)[0]
            Queue = [u]
            dist = {u: 0}
            marked[u] = True
            dist_to_cover[u] = 0
            _L.append(u)
            num_marked+=1
            while len(Queue):
                v = Queue.pop(0)
                for nbr_v in G.neighbors(v):
                    dist_nv = dist[v] + 1
                    if nbr_v not in dist:
                        if not marked[nbr_v] and dist_nv <= epsilon:
                            num_marked+=1
                            marked[nbr_v] = True
                            dist_to_cover[nbr_v] = dist_nv
                        dist[nbr_v] = dist_nv
                        Queue.append(nbr_v)
            # print(len(_L),' ',num_marked)
            for v in G.nodes:
                if not marked[v]:
                    u = v
                    break
        return _L, dist_to_cover

# compute the |L| x |L| matrix => sparse distance metric on landmarks
def get_sparse_matrix(G,dist_to_cover,landmarks):
    data = {}
    all_pairSP_len = dict(nx.all_pairs_shortest_path_length(G))
    for i,u in enumerate(landmarks):
        for j,v in enumerate(landmarks):
            if i<j:
                e_ij = math.inf
                for n in G.nodes:
                    # print(u,v,n)
                    dist_i = all_pairSP_len[u].get(n,math.inf)
                    dist_j = all_pairSP_len[v].get(n,math.inf)
                    mx = max( max(dist_i,dist_j) - dist_to_cover[n],0.0)
                    if mx < e_ij:
                        e_ij = mx
                data[(i,j)] = e_ij
            elif j==i:
                data[(i,j)] = 0.0
            else:
                data[(i,j)] = data[(j,i)]
    I,J,D,INFINITY = [],[],[],0
    for key,val in data.items():
        I.append(key[0])
        J.append(key[1])
        D.append(val)
        if val != math.inf:
            INFINITY = max(val,INFINITY)
    del data
    N = len(landmarks)
    return sparse.coo_matrix((D, (I, J)), shape=(N, N)).tocsr(), INFINITY
