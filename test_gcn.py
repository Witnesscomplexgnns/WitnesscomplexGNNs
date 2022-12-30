import torch
import numpy as np
import torch.nn.functional as F
from deeprobust.graph.defense import GCN, WitCompNN
from deeprobust.graph.utils import *
from deeprobust.graph.data import Dataset
import scipy.sparse as sp
import argparse
from utils import load_npz
import gudhi as gd
import json

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=15, help='Random seed.')
parser.add_argument('--dataset', type=str, default='cora', choices=['cora'], help='dataset')
parser.add_argument('--lr', type=float, default=0.01,  help='learning rate')
parser.add_argument('--drop_rate', type=float, default=0.5,  help='dropout rate')
parser.add_argument('--weight_decay', type=float, default=5e-4,  help='weight decay rate')
parser.add_argument('--ptb_rate', type=float, default=0.05,  help='pertubation rate')
parser.add_argument('--epoch', type=float, default=500,  help='epochs')

args = parser.parse_args()
args.cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# load original data and attacked data
adj, features, labels = load_npz('data/' + args.dataset + '/' + args.dataset + '.npz')
f = open('data/' + args.dataset + '/' + args.dataset + '_prognn_splits.json')
idx = json.load(f)
idx_train, idx_val, idx_test = np.array(idx['idx_train']), np.array(idx['idx_val']), np.array(idx['idx_test'])
perturbed_adj = sp.load_npz('data/' + args.dataset + '/' + args.dataset + '_meta_adj_0.05.npz')

np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)


# Setup GCN Model
model = GCN(nfeat=features.shape[1], nhid=16, nclass=int(labels.max())+1, dropout=args.drop_rate, lr=args.lr, weight_decay=args.weight_decay, device=device)
model = model.to(device)
model.fit(features, perturbed_adj, labels, idx_train, train_iters=args.epoch, verbose=True)
# # using validation to pick model
# model.fit(features, perturbed_adj, labels, idx_train, idx_val, train_iters=200, verbose=True)
model.eval()
# You can use the inner function of model to test
model.test(idx_test)
