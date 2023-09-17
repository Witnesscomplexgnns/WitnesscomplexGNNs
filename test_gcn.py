import torch
import numpy as np
import torch.nn.functional as F
from deeprobust.graph.defense import GCN
from deeprobust.graph.utils import *
from deeprobust.graph.data import Dataset
import scipy.sparse as sp
import argparse
from utils import load_npz
# import gudhi as gd
import json,os
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=15, help='Random seed.')
parser.add_argument('--dataset', type=str, default='cora', choices=['cora','citeseer'], help='dataset')
parser.add_argument('--lr', type=float, default=0.001,  help='learning rate')
parser.add_argument('--drop_rate', type=float, default=0.5,  help='dropout rate')
parser.add_argument('--weight_decay', type=float, default=5e-4,  help='weight decay rate')
parser.add_argument('--ptb_rate', type=float, default=0.05,  help='pertubation rate')
parser.add_argument('--epoch', type=float, default=200,  help='epochs')

args = parser.parse_args()
print(args)
args.cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# load original data and attacked data
adj, features, labels = load_npz('data/' + args.dataset + '/' + args.dataset + '.npz')
f = open('data/' + args.dataset + '/' + args.dataset + '_prognn_splits.json')
idx = json.load(f)
idx_train, idx_val, idx_test = np.array(idx['idx_train']), np.array(idx['idx_val']), np.array(idx['idx_test'])
if args.ptb_rate == 0:
    print('loading unpurturbed adj')
    perturbed_adj,_,_ = load_npz('data/' + args.dataset + '/' + args.dataset + '.npz')
else:
    print('loading perturbed adj. ')
    perturbed_adj = sp.load_npz('data/' + args.dataset + '/' + args.dataset + '_meta_adj_'+str(args.ptb_rate)+'.npz')

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
acc = model.test(idx_test)
output = {'seed':args.seed,'acc':acc}
csv_name = 'GCN_acc_'+args.dataset + "_" + str(args.ptb_rate) + '.csv'
if os.path.exists(csv_name):
    result_df = pd.read_csv(csv_name)
else:
    result_df = pd.DataFrame()
result = pd.concat([result_df, pd.DataFrame(output,index = [0])])
result.to_csv(csv_name, header=True, index=False)
print(result.head(10))
print('Mean=> ',result['acc'].mean(),' std => ',result['acc'].std())