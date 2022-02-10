import torch
import torch.nn as nn
import torch.nn.functional as F


class Classifier(nn.Module):
    def __init__(self, in_features, out_features):
        super(Classifier, self).__init__()

        self.fc1 = nn.Linear(in_features, 1024)
        self.bn1 = nn.BatchNorm1d(1024)  #BatchNorm1d-->LayerNorm
        self.fc2 = nn.Linear(1024, 512)
        self.bn2 = nn.BatchNorm1d(512)
        self.fc3 = nn.Linear(512, 256)
        self.bn3 = nn.BatchNorm1d(256)
        self.fc4 = nn.Linear(256, out_features)
        self.dropout = nn.Dropout(p=0.3)

    def forward(self, x):

        x = self.dropout(F.leaky_relu(self.bn1(self.fc1(x))))
        x = self.dropout(F.leaky_relu(self.bn2(self.fc2(x))))
        x = self.dropout(F.leaky_relu(self.bn3(self.fc3(x))))
        x = torch.sigmoid(self.fc4(x))

        return x

class Classifier_new(nn.Module):
    # __constants__ = ['in_features', 'out_features']
    def __init__(self, in_features, out_features):
        super(Classifier_new, self).__init__()

        self.fc1 = nn.Linear(in_features, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.fc3 = nn.Linear(128, 32)
        self.bn3 = nn.BatchNorm1d(32)
        self.fc4 = nn.Linear(32, out_features)
        self.dropout = nn.Dropout(p=0.3)

    def forward(self, x):

        x = self.dropout(F.leaky_relu(self.bn1(self.fc1(x))))
        x = self.dropout(F.leaky_relu(self.bn2(self.fc2(x))))
        x = self.dropout(F.leaky_relu(self.bn3(self.fc3(x))))
        x = torch.sigmoid(self.fc4(x))

        return x

class Classifier_change(nn.Module):
    # __constants__ = ['in_features', 'out_features']
    def __init__(self, in_features, out_features):
        super(Classifier_change, self).__init__()

        self.fc1 = nn.Linear(in_features, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, 64)
        self.bn2 = nn.BatchNorm1d(64)
        self.fc3 = nn.Linear(64, 16)
        self.bn3 = nn.BatchNorm1d(16)
        self.fc4 = nn.Linear(16, out_features)
        self.dropout = nn.Dropout(p=0.3)

    def forward(self, x):

        x = self.dropout(F.leaky_relu(self.bn1(self.fc1(x))))
        x = self.dropout(F.leaky_relu(self.bn2(self.fc2(x))))
        x = self.dropout(F.leaky_relu(self.bn3(self.fc3(x))))
        x = torch.sigmoid(self.fc4(x))

        return x

class Classifier_down(nn.Module): #E3-subsarate
    # __constants__ = ['in_features', 'out_features']
    def __init__(self, in_features, out_features, hidden_size=800):
        super(Classifier_down, self).__init__()
        self.project = nn.Sequential(
            nn.Linear(out_features, hidden_size),  # 定义全连接层 64,128
            nn.Tanh(),  # 定义激活函数 Tanh
            nn.Linear(hidden_size, 1, bias=False)  # 每一个节点在metapath下的值1维   # 猜测为重要性
        )
        self.dnn1 = nn.Sequential(
            nn.Linear(in_features, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(256, out_features),
            nn.Sigmoid()
        )
        self.dnn2 = nn.Sequential(
            nn.Linear(in_features, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(256, out_features),
            nn.Sigmoid()
        )

    def forward(self, x):
        x1 = self.dnn1(x[:, 0, :])
        x2 = self.dnn2(x[:, 1, :])

        x = torch.stack((x1, x2), dim=1)

        w = self.project(x).mean(0)
        exbeta = torch.softmax(w, dim=0)  # (M, 1) 归一化 每一个metapath下的权重
        beta = exbeta.expand((x.shape[0],) + exbeta.shape)  # (N, M, 1)     [12107, 2, 1]
        x = (beta * x).sum(1)                           # [12107, 2, 1] * [12107, 2, 400]

        return x, exbeta


class Classifier_mul(nn.Module):
    # __constants__ = ['in_features', 'out_features']
    def __init__(self, in_features, out_features, hidden_size=800):
        super(Classifier_mul, self).__init__()
        self.project = nn.Sequential(
            nn.Linear(out_features, hidden_size),  # 定义全连接层 64,128
            nn.Tanh(),  # 定义激活函数 Tanh
            nn.Linear(hidden_size, 1, bias=False)  # 每一个节点在metapath下的值1维   # 猜测为重要性
        )
        self.dnn1 = nn.Sequential(
            nn.Linear(in_features, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(256, int(out_features/2)),
            nn.Sigmoid()
        )
        self.dnn2 = nn.Sequential(
            nn.Linear(in_features, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(256, int(out_features/2)),
            nn.Sigmoid()
        )
        self.dnn3 = nn.Sequential(
            nn.Linear(in_features, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(256, int(out_features/2)),
            nn.Sigmoid()
        )
        self.dnn4 = nn.Sequential(
            nn.Linear(in_features, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(256, int(out_features/2)),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x1 = torch.from_numpy(x[:, 0, :].squeeze()).float()
        # x2 = torch.from_numpy(x[:, 1, :].squeeze()).float()
        # x3 = torch.from_numpy(x[:, 2, :].squeeze()).float()
        # x4 = torch.from_numpy(x[:, 3, :].squeeze()).float()

        x1 = self.dnn1(x[:, 0, :])
        x2 = self.dnn2(x[:, 1, :])
        x3 = self.dnn3(x[:, 0, :])
        x4 = self.dnn4(x[:, 1, :])
        # x3 = self.dnn3(x3)
        # x4 = self.dnn4(x4)

        ppi = torch.cat((x1,x3),dim=0)
        ssn = torch.cat((x2,x4),dim=0)
        x = torch.stack((ppi, ssn), dim=1)

        w = self.project(x).mean(0)
        exbeta = torch.softmax(w, dim=0)  # (M, 1) 归一化 每一个metapath下的权重
        beta = exbeta.expand((x.shape[0],) + exbeta.shape)  # (N, M, 1)     [12107, 2, 1]
        x = (beta * x).sum(1)  # [12107, 2, 1] * [12107, 2, 400]

        return x, exbeta