import torch
from torch import nn, optim

from model import Classifier,Classifier_new


def train_data(X_train, Y_train, X_test, epochs):

    model = Classifier_new(int(X_train.shape[-1]), Y_train.shape[-1])
    X_train = torch.from_numpy(X_train).float()
    Y_train = torch.from_numpy(Y_train).float()
    X_test = torch.from_numpy(X_test).float()

    loss_fcn = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    for e in range(epochs):

        model.train()
        logits = model(X_train)
        loss = loss_fcn(logits, Y_train)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print('Epoch {:d} | Train Loss {:.4f}'.format(
            e + 1, loss.item()))

    model.eval()

    y_prob = model(X_test)

    torch.save(model, "model.pkl")

    return y_prob



def train_model(X_train, Y_train, epochs):

    model = Classifier(int(X_train.shape[-1]), Y_train.shape[1])

    X_train = torch.from_numpy(X_train).float()
    Y_train = torch.from_numpy(Y_train).float()

    loss_fcn = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(),lr = 0.0001)

    for e in range(epochs):

        model.train()
        logits= model(X_train)
        loss = loss_fcn(logits, Y_train)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print('Epoch {:d} | Train Loss {:.4f}'.format(
            e + 1, loss.item()))

    torch.save(model, "model.pkl")

    return

