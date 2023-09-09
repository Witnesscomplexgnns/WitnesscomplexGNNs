import torch
import numpy as np
import torch.nn.functional as F
from deeprobust.graph.defense import GCN, GWitCompNN, LWitCompNN_V1, LWitCompNN_V2, LWitCompNN_V3
from deeprobust.graph.utils import *
from deeprobust.graph.data import Dataset
import scipy.sparse as sp
import argparse
from utils import load_npz
#import gudhi as gd
import json

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=15, help='Random seed.')
parser.add_argument('--dataset', type=str, default='cora', choices=['cora'], help='dataset')
parser.add_argument('--lr', type=float, default=0.01,  help='learning rate')
parser.add_argument('--drop_rate', type=float, default=0.5,  help='dropout rate')
parser.add_argument('--weight_decay', type=float, default=5e-4,  help='weight decay rate')
parser.add_argument('--ptb_rate', type=float, default=0.05,  help='pertubation rate')
parser.add_argument('--epoch', type=int, default=500,  help='epochs')
parser.add_argument('--alpha', type=float, default=0.3,  help='alpha')
parser.add_argument('--beta', type=float, default=0.7,  help='beta')
# parser.add_argument('--topo', type=str,default = 'witptb_local',help='witorig/witptb/witptb_local/vrorig/vrptb')
parser.add_argument('--topo', type=str,default = 'local',help='local/global')
parser.add_argument('--method',type=str,default='transformer',help='transformer/resnet/cnn')

args = parser.parse_args()
args.cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# load original data and attacked data
adj, features, labels = load_npz('data/' + args.dataset + '/' + args.dataset + '.npz')
f = open('data/' + args.dataset + '/' + args.dataset + '_prognn_splits.json')
idx = json.load(f)
idx_train, idx_val, idx_test = np.array(idx['idx_train']), np.array(idx['idx_val']), np.array(idx['idx_test'])
perturbed_adj = sp.load_npz('data/' + args.dataset + '/' + args.dataset + '_meta_adj_'+str(args.ptb_rate)+'.npz')

np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

# Setup WitCompNN Model        
# load witness complex topological features
# if args.topo == 'witorig':
#     witness_complex_feat = torch.FloatTensor(np.load('data/' + args.dataset + '/' + args.dataset + '_PI' + '.npz', allow_pickle=True)['arr_0'])

# if args.topo == 'witptb':
if args.topo == 'global': # load Global PI computed on the perturbed Adj matrix.
    # witness_complex_feat = torch.FloatTensor(np.random.rand(1,50,50)) #torch.FloatTensor(np.load('data/' + args.dataset + '/' + args.dataset + '_'+str(args.ptb_rate)+'_PI' + '.npz', allow_pickle=True)['arr_0'])
    witness_complex_feat = torch.FloatTensor(np.load('data/' + args.dataset + '/' + args.dataset + '_'+str(args.ptb_rate)+'_PI' + '.npz', allow_pickle=True)['arr_0'])
    print('shape of local PI representation: ',witness_complex_feat.shape)

elif args.topo == 'local': # Load Local PI computed on the perturbed Adj matrix.
    # local_witness_complex_feat => Shape (#nodes x 50 x 50)
    tmp = np.load('data/' + args.dataset + '/' + args.dataset + '_'+str(args.ptb_rate)+'_localPI' + '.npz', allow_pickle=True)['arr_0']
    # print('tmp.shape: ',tmp.shape)
    # local_witness_complex_feat_ = np.random.rand(2485, 50, 50) # torch.FloatTensor(np.load('data/' + args.dataset + '/' + args.dataset + '_'+str(args.ptb_rate)+'_localPI' + '.npz', allow_pickle=True)['arr_0'])
    local_witness_complex_feat_ = np.expand_dims(tmp, axis=1)  # (#nodes, 1, pi_dim, pi_dim)
    local_witness_complex_feat = torch.FloatTensor(local_witness_complex_feat_)
    print('shape of local PI representation: ',local_witness_complex_feat.shape) # (#nodes, 1, pi_dim, pi_dim)
else:
    raise ValueError("Wrong args.topo")

topo_type = args.topo # 'local', 'global'
method = args.method
aggregation_method = 'weighted_sum' # einsum, weighted_sum, attention

if topo_type == 'global':
    model = GWitCompNN(nfeat=features.shape[1], nhid=16, nclass=int(labels.max()) + 1, dropout=args.drop_rate,  lr=args.lr, weight_decay=args.weight_decay, device=device)
    model = model.to(device)
    model.fit(features, perturbed_adj, witness_complex_feat, labels, idx_train, train_iters=args.epoch, verbose=True)
else:
    if method == 'resnet':
        print("You are using ResNet now!")
        model = LWitCompNN_V1(nfeat=features.shape[1], nhid=16, nclass=int(labels.max()) + 1, dropout=args.drop_rate, lr=args.lr, weight_decay=args.weight_decay, device=device, alpha = args.alpha, beta = args.beta, aggregation_method = aggregation_method)
        model = model.to(device)
        model.fit(features, perturbed_adj, local_witness_complex_feat, labels, idx_train, train_iters=args.epoch,  verbose=True)
    elif method == 'cnn':
        print("You are using CNN now!")
        model = LWitCompNN_V2(nfeat=features.shape[1], nhid=16, nclass=int(labels.max()) + 1, dropout=args.drop_rate, lr=args.lr, weight_decay=args.weight_decay, device=device, alpha = args.alpha, beta = args.beta, aggregation_method = aggregation_method)
        model = model.to(device)
        model.fit(features, perturbed_adj, local_witness_complex_feat, labels, idx_train, train_iters=args.epoch, verbose=True)
    elif method == 'transformer':
        print("You are using Transformer now!")
        model = LWitCompNN_V3(nfeat=features.shape[1], nhid=16, nclass=int(labels.max()) + 1, dropout=args.drop_rate, lr=args.lr, weight_decay=args.weight_decay, device=device, alpha = args.alpha, beta = args.beta, aggregation_method = aggregation_method)
        model = model.to(device)
        model.fit(features, perturbed_adj, local_witness_complex_feat, labels, idx_train, train_iters=args.epoch, verbose=True)

# # using validation to pick model
# model.fit(features, perturbed_adj, labels, idx_train, idx_val, train_iters=200, verbose=True)
model.eval()
# You can use the inner function of model to test
model.test(idx_test)





# local_witness_complex_feat_ = np.expand_dims(local_witness_complex_feat_, axis=1)  # (#nodes, 1, pi_dim, pi_dim)
# local_witness_complex_feat = np.tile(local_witness_complex_feat_, (1, 3, 1, 1))  # (#nodes, 3, pi_dim, pi_dim) which can be fed into resnet
# local_witness_complex_feat = torch.FloatTensor(local_witness_complex_feat)