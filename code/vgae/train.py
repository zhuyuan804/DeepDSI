from __future__ import division
from __future__ import print_function

import time

import numpy as np
import scipy.sparse as sp
import torch
from torch import optim
from torch.autograd import Variable
from .model import GCNModelVAE,VAE
from .optimizer import vae_loss_function,loss_function
from .utils import preprocess_graph, sparse_to_tuple
from visdom import Visdom



def train_vgae(features, adj_train, args, epochs):
    n_nodes, feat_dim = features.shape
    # features = sparse_to_tuple((features.tocoo()))

    # Store original adjacency matrix (without diagonal entries) for later
    adj_orig = adj_train
    adj_orig = adj_orig - sp.dia_matrix((adj_orig.diagonal()[np.newaxis, :], [0]), shape=adj_orig.shape)
    adj_orig.eliminate_zeros()

    adj = adj_train     #8976train

    # Some preprocessing
    adj_norm = preprocess_graph(adj)

    adj_label = adj_train + sp.eye(adj_train.shape[0]) #添加单位矩阵
    # adj_label = sparse_to_tuple(adj_label)
    features = torch.FloatTensor(features.toarray())
    adj_label = torch.FloatTensor(adj_label.toarray())

    pos_weight = torch.Tensor([float(adj.shape[0] * adj.shape[0] - adj.sum()) / adj.sum()])
    norm = adj.shape[0] * adj.shape[0] / float((adj.shape[0] * adj.shape[0] - adj.sum()) * 2)

    model = GCNModelVAE(feat_dim, args.hidden1, args.hidden2, args.dropout)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    hidden_emb = None

    loss_all = []
    for epoch in range(epochs):
        t = time.time()
        model.train()
        optimizer.zero_grad()
        recovered, z, mu, logvar = model(features, adj_norm)
        loss = loss_function(preds=recovered, labels=adj_label,
                             mu=mu, logvar=logvar, n_nodes=n_nodes,
                             norm=norm, pos_weight=pos_weight)
        loss.backward()
        cur_loss = loss.item()
        optimizer.step()

        hidden_emb = mu.data.numpy()

        if epoch % 10 == 0:
            print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(cur_loss),"time=", "{:.5f}".format(time.time() - t))

    print("Optimization Finished!")

    return hidden_emb

def normalize(data):
    _range = np.max(data) - np.min(data)
    return (data - np.min(data)) / _range




