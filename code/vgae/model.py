import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import GraphConvolution
from torch.autograd import Variable

class GCNModelVAE(nn.Module):
    def __init__(self, input_feat_dim, hidden_dim1, hidden_dim2, dropout):
        super(GCNModelVAE, self).__init__()
        self.gc1 = GraphConvolution(input_feat_dim, hidden_dim1, dropout, act=F.relu)
        self.gc2 = GraphConvolution(hidden_dim1, hidden_dim2, dropout, act=lambda x: x)
        self.gc3 = GraphConvolution(hidden_dim1, hidden_dim2, dropout, act=lambda x: x)
        self.dc = InnerProductDecoder(dropout, act=lambda x: x)

    def encode(self, x, adj):       # 两次GCN求embedding，一个作为均值，一个作为方差（正态分布的）
        hidden1 = self.gc1(x, adj)
        return self.gc2(hidden1, adj), self.gc3(hidden1, adj)

    def reparameterize(self, mu, logvar):       # 通过均值和方差得到z
        if self.training:
            std = torch.exp(logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu

    def forward(self, x, adj):      # 2708*1433     2708*2708
        mu, logvar = self.encode(x, adj)
        z = self.reparameterize(mu, logvar)     # 从分布中采样重构得到z
        return self.dc(z),z, mu, logvar


class InnerProductDecoder(nn.Module):
    """Decoder for using inner product for prediction."""

    def __init__(self, dropout, act=torch.sigmoid):
        super(InnerProductDecoder, self).__init__()
        self.dropout = dropout
        self.act = act

    def forward(self, z):
        z = F.dropout(z, self.dropout, training=self.training)
        adj = self.act(torch.mm(z, z.t()))      # 2708*2708 利用内积重构原始的adj
        return adj


class VAE(nn.Module):
    def __init__(self,input_feat_dim,hidden_dim1,hidden_dim2,dropout):
        super(VAE, self).__init__()

        self.fc1 = nn.Linear(input_feat_dim, hidden_dim1,dropout)
        self.fc21 = nn.Linear(hidden_dim1, hidden_dim2,dropout)
        self.fc22 = nn.Linear(hidden_dim1, hidden_dim2,dropout)
        self.fc3 = nn.Linear(hidden_dim2, hidden_dim1)
        self.fc4 = nn.Linear(hidden_dim1,input_feat_dim)
        self.dc = InnerProductDecoder(dropout, act=lambda x: x)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        eps = torch.FloatTensor(std.size()).normal_()
        eps = Variable(eps)
        return eps.mul(std).add_(mu)

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        # return F.sigmoid(self.fc4(h3))
        return torch.sigmoid(self.fc4(h3))

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparametrize(mu, logvar)
        return self.dc(z),self.decode(z), mu, logvar

