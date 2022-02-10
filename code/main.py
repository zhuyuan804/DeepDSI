import numpy as np
import pandas as pd
import pickle as pkl
import argparse
import json
import os
from sklearn.model_selection import KFold
import torch
from utils import load_data
from train import train_data,train_model
from utils import generate_data, evaluate_performance,plot_roc,predict_new_dsi
from vgae.train import train_vgae

def train(args):

    print("Import embed vector")

    uniprot = pd.read_pickle(os.path.join(args.data_path, args.species, "data/processing/feature.pkl"))
    adj, features = load_data(uniprot, args)

    embeddings = train_vgae(features, adj, args, args.epochs)

    #Save the embedding vector
    # path = os.path.join(args.data_path, args.species, "output/graph_embeddings_vector.pkl")
    #
    # with open(path, 'wb') as file:
    #     pkl.dump(embeddings, file)

    #read the embedding vector

    # embeddings = pd.read_pickle(os.path.join(args.data_path, args.species + "/output/vgae/graph_embeddings_vector.pkl"))

    np.random.seed(5959)
    #The training dataset was before 2018.1, and the independent test dataset was after 2018.1
    gsp_train_file = os.path.join(args.data_path, args.species, "networks/gsp_train.txt")
    gsp_train = pd.read_table(gsp_train_file, delimiter="\t")
    gsp_train = np.array(gsp_train)
    gsp_test_file = os.path.join(args.data_path, args.species, "networks/gsp_test.txt")
    gsp_test = pd.read_table(gsp_test_file, delimiter="\t")
    gsp_test = np.array(gsp_test)
    gsn_train_file = os.path.join(args.data_path, args.species, "networks/gsn_train.txt")
    gsn_train = pd.read_table(gsn_train_file, delimiter="\t")
    gsn_train = np.array(gsn_train)
    gsn_test_file = os.path.join(args.data_path, args.species, "networks/gsn_test.txt")
    gsn_test = pd.read_table(gsn_test_file, delimiter="\t")
    gsn_test = np.array(gsn_test)

    X, Y = generate_data(embeddings, gsp_train, gsn_train, args)

    index = np.concatenate([gsp_train, gsn_train], axis=0)

    print("The" + str(args.folds) + "fold cross validation")
    rs = KFold(n_splits=args.folds, shuffle=True)
    all_idx = list(range(X.shape[0]))
    cv_index_set = rs.split(all_idx)  # Five-fold cross validation and independent testing
    np.random.shuffle(all_idx)
    all_X = X[all_idx]
    all_Y = Y[all_idx]
    index = index[all_idx]
    all_Y_label = []
    all_Y_pred = []
    all_index = []
    fold = 1

    for train_idx, test_idx in cv_index_set:
        np.random.shuffle(train_idx)
        np.random.shuffle(test_idx)
        X_train = X[train_idx]
        X_test = X[test_idx]
        train_label = Y[train_idx]
        test_label = Y[test_idx]
        test_index = index[test_idx]

        print("###################################")
        print("The" + str(fold) + "cross validation is underway")
        fold = fold + 1

        Y_pred = train_data(X_train, train_label, X_test, args.epochs)

        Y_pred = Y_pred.data.numpy()
        all_Y_label.append(test_label)
        all_Y_pred.append(Y_pred)
        all_index.append(test_index)

    Y_label = np.concatenate(all_Y_label)
    Y_pred = np.concatenate(all_Y_pred)
    Y_pred = Y_pred.reshape(Y_pred.shape[0], 1)
    index_fold = np.concatenate(all_index)

    plot_roc(Y_label, Y_pred, '5-CV')

    perf = evaluate_performance(Y_label, Y_pred)

    def output_data(Y_label, Y_pred, str, perf):
        with open(os.path.join(args.save_path, 'Y_label_' + str + '.pkl'), 'wb') as file:
            pkl.dump(Y_label, file)
        with open(os.path.join(args.save_path, 'Y_pred_' + str + '.pkl'), 'wb') as file:
            pkl.dump(Y_pred, file)

        # evaluate_performance_per(Y_label, Y_pred, index_fold)
        if args.save_results:
            with open(os.path.join(args.save_path, str + ".json"), "w+") as f:
                json.dump(perf, f)

    output_data(Y_label, Y_pred, 'gsd', perf)
    # ############################################################################
    #
    train_model(all_X, all_Y, args.epochs)

    #############################################################################

    model = torch.load("model.pkl")
    X_, Y_ = generate_data(embeddings, gsp_test, gsn_test, args)

    # index_ind = np.concatenate([ind, gsn_test], axis=0)

    X_ = torch.from_numpy(X_).float()
    prediction = model(X_)
    prediction = prediction.data.numpy()

    perf = evaluate_performance(Y_, prediction)

    plot_roc(Y_, prediction, 'independent test')

    output_data(Y_, prediction, 'independent', perf)

    ####################################################################
    positive = np.concatenate((gsp_train, gsp_test), axis=0)
    negative = np.concatenate((gsn_train, gsn_test), axis=0)
    X, Y = generate_data(embeddings, positive, negative, args)
    all_idx = list(range(X.shape[0]))
    np.random.shuffle(all_idx)

    all_X = X[all_idx]
    all_Y = Y[all_idx]
    train_model(all_X, all_Y, args.epochs)

    print("Predict new DSI")
    pred_dsi = predict_new_dsi(embeddings, gsp_train, gsp_test, args)
    pred_dsi.to_csv(os.path.join(args.save_path, 'predict_dsi.csv'))

    print('The end')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # Global parameters
    parser.add_argument('--graphs', type=lambda s: [item for item in s.split(",")], default=['combined', 'similarity'],help="lists of graphs to use.")
    parser.add_argument('--species', type=str, default="human", help="which species to use.")
    parser.add_argument('--data_path', type=str, default="../", help="path storing data.")
    parser.add_argument('--save_path', type=str, default="../human/output", help="path save output data")
    parser.add_argument('--save_results', type=int, default=1, help="whether to save the performance results")
    # GCN training parameters
    parser.add_argument('--lr', type=float, default=0.0001, help="Initial learning rate.")
    parser.add_argument('--hidden1', type=int, default=800, help="Number of units in hidden layer 1.")
    parser.add_argument('--hidden2', type=int, default=400, help="Number of units in hidden layer 2.")
    parser.add_argument('--dropout', type=float, default=0.1, help="Dropout rate (1 - keep probability).")

    # DNN training parameters
    parser.add_argument('--folds', type=int, default=5, help="Number of folds.")
    parser.add_argument('--epochs', type=int, default=120, help="Number of epochs to train classifier.")


    args = parser.parse_args()
    print(args)
    train(args)


