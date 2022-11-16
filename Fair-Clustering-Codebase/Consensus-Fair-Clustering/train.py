from __future__ import division
from __future__ import print_function
import random
import time
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

from models import GMLP, ClusteringLayer
from utils import get_A_r, sparse_mx_to_torch_sparse_tensor, target_distribution, aff
import warnings
warnings.filterwarnings('ignore')

from scipy import sparse
from torch import nn
from eval import *

from sklearn.cluster import KMeans

# Settings
parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--epochs', type=int, default=400,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.01,
                    help='learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=256,
                    help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.6,
                    help='Dropout rate (1 - keep probability).')
parser.add_argument('--data', type=str, default='cora',
                    help='dataset to be used')
parser.add_argument('--alpha', type=float, default=2.0,
                    help='To control the ratio of Ncontrast loss')
parser.add_argument('--beta', type=float, default=2.0,
                    help='To control the ratio of partition loss')
parser.add_argument('--batch_size', type=int, default=2048,
                    help='batch size')
parser.add_argument('--order', type=int, default=2,
                    help='to compute order-th power of adj')
parser.add_argument('--tau', type=float, default=1.0,
                    help='temperature for Ncontrast loss')
parser.add_argument('--k', type=int, default=10,
                    help='Number of clusters')


args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()



## get data
adj, features, labels = np.load('datafiles/' + args.data + '_coassoc.npy'),  np.load('datafiles/' + args.data + '_X.npy'),  np.load('datafiles/' + args.data + '_y.npy')
adj = sparse.csr_matrix(adj)
adj = sparse_mx_to_torch_sparse_tensor(adj).float()
features = torch.FloatTensor(features).float()
labels = torch.LongTensor(labels)
idx_train = np.array(range(len(features)))
idx_train = torch.LongTensor(idx_train)

adj_label = get_A_r(adj, args.order)

adj, adj_label, features, idx_train = adj.cuda(), adj_label.cuda(), features.cuda(), idx_train.cuda()


s = np.load('datafiles/' + args.data + '_s.npy')
s_idx0, s_idx1 = [], []
for i in range(len(s)):
    if s[i] == 0:
        s_idx0.append(i)
    elif s[i] == 1:
        s_idx1.append(i) 


L = np.load('precomputed_labels/labels_' + args.data + '.npy')
Y = np.zeros((len(s), args.k))
for i,l in enumerate(L):
    Y[i,l] = 1.0
Y = torch.FloatTensor(Y).float().cuda()

MSEL = nn.MSELoss(reduction="sum")

## Model and optimizer
model = GMLP(nfeat=features.shape[1],
            nhid=args.hidden,
            nclass=labels.max().item() + 1,
            dropout=args.dropout,
            )


CL = ClusteringLayer(cluster_number=args.k, hidden_dimension=args.hidden).cuda()

optimizer = optim.Adam(model.get_parameters() + CL.get_parameters(),
                       lr=args.lr, weight_decay=args.weight_decay)

KL_div = nn.KLDivLoss(reduction="sum")


if args.cuda:
    model.cuda()
    features = features.cuda()
    labels = labels.cuda()
    idx_train = idx_train.cuda()


def Ncontrast(x_dis, adj_label, tau = 1):
    """
    compute the Ncontrast loss
    """
    x_dis = torch.exp( tau * x_dis)
    x_dis_sum = torch.sum(x_dis, 1)
    x_dis_sum_pos = torch.sum(x_dis*adj_label, 1)
    loss = -torch.log(x_dis_sum_pos * (x_dis_sum**(-1))+1e-8).mean()
    return loss

def get_batch(batch_size):
    """
    get a batch of feature & adjacency matrix
    """
    rand_indx = torch.tensor(np.random.choice(np.arange(adj_label.shape[0]), batch_size)).type(torch.long).cuda()
    rand_indx[0:len(idx_train)] = idx_train
    features_batch = features[rand_indx]
    adj_label_batch = adj_label[rand_indx,:][:,rand_indx]
    return features_batch, adj_label_batch

def train():
    features_batch, adj_label_batch = get_batch(batch_size=args.batch_size)
    model.train()

    CL.train()

    optimizer.zero_grad()
    output, x_dis, embeddings = model(features_batch)

    output = CL(embeddings)
    output0, output1 = output[s_idx0], output[s_idx1]
    target0, target1 = target_distribution(output0).detach(), target_distribution(output1).detach()
    fair_loss = 0.5 * KL_div(output0.log(), target0) + 0.5 * KL_div(output1.log(), target1)

    loss_Ncontrast = Ncontrast(x_dis, adj_label_batch, tau = args.tau)

    predict0, predict1 = Y[s_idx0], Y[s_idx1]
    partition_loss = 0.5 * MSEL(aff(output0), aff(predict0)) + 0.5 * MSEL(aff(output1), aff(predict1))

    loss_train = args.alpha * fair_loss + loss_Ncontrast + args.beta * partition_loss ###

    loss_train.backward()
    optimizer.step()
    return 

def test():
    model.eval()
    logits, embeddings = model(features)

    CL.eval()
    preds = CL(embeddings)
    preds = preds.cpu().detach().numpy()
    pred_labels = np.argmax(preds, axis=1)
    bal = balance(pred_labels, features.cpu(), s)
    ent = entropy(pred_labels, s)
    NMI = nmi(labels.cpu(), pred_labels)
    ACC = acc(labels.cpu(), pred_labels)
    return (bal, ent, NMI, ACC)


for epoch in tqdm(range(args.epochs)):
    train()

torch.save(model, 'saved_models/' + args.data + '-GMLP.pt')
torch.save(CL, 'saved_models/' + args.data + '-CL.pt')

model = torch.load('saved_models/' + args.data + '-GMLP.pt')
CL = torch.load('saved_models/' + args.data + '-CL.pt')

model.eval()
logits, embeddings = model(features)

CL.eval()
preds = CL(embeddings)
preds = preds.cpu().detach().numpy()
pred_labels = np.argmax(preds, axis=1)
print("balance: {}".format(balance(pred_labels, features.cpu(), s)))
print("entropy: {}".format(entropy(pred_labels, s)))
print("nmi: {}".format(nmi(labels.cpu(), pred_labels)))
print("acc: {}".format(acc(labels.cpu(), pred_labels)))

