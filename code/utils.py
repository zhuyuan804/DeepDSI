import os
from tqdm import tqdm
import numpy as np
import pandas as pd
import torch
import scipy.sparse as sp
from sklearn.preprocessing import scale
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from collections import defaultdict
from sklearn.metrics import roc_curve,auc
import matplotlib.pyplot as plt

#############   load data  #######################

def reshape(features):
    return np.hstack(features).reshape((len(features), len(features[0])))


def load_ssn_network(filename, gene_num):
    print('Import SSN network')
    with open(filename) as f:
        data = f.readlines()
    adj = np.zeros((gene_num,gene_num))
    for x in tqdm(data):
        temp = x.strip().split("\t")
        adj[int(temp[0]),int(temp[1])] = 1
    if (adj.T == adj).all():
        pass
    else:
        adj = adj + adj.T

    return adj

def load_data(uniprot, args):

    features_seq = scale(reshape(uniprot['features_seq'].values)) #Data standardization
    features = features_seq
    features = sp.csr_matrix(features)
    filename = os.path.join(args.data_path, args.species , "data/processing/sequence_similar_network.txt")
    adj = load_ssn_network(filename, uniprot.shape[0])

    adj = sp.csr_matrix(adj)

    return adj, features

#############   evaluation  #######################

def precision_max(preds, labels):
    preds = np.round(preds, 2)
    labels = labels.astype(np.int32)
    f_max = 0
    p_max = 0
    r_max = 0
    t_max = 0
    for t in range(1, 100):
        threshold = t / 100.0
        predictions = (preds > threshold).astype(np.int32)
        p0 = (preds < threshold).astype(np.int32)
        tp = np.sum(predictions * labels)
        fp = np.sum(predictions) - tp
        fn = np.sum(labels) - tp
        tn = np.sum(p0) - fn
        sn = tp / (1.0 * np.sum(labels))
        sp = np.sum((predictions ^ 1) * (labels ^ 1))
        sp /= 1.0 * np.sum(labels ^ 1)
        fpr = 1 - sp
        precision = tp / (1.0 * (tp + fp))
        recall = tp / (1.0 * (tp + fn))
        f = 2 * precision * recall / (precision + recall)
        acc = (tp + tn) / (tp + fp + tn + fn)
        if p_max < precision:
            f_max = f
            a_max = acc
            p_max = precision
            r_max = recall
            sp_max = sp
            t_max = threshold
    return f_max, a_max, p_max, r_max, t_max

def calculate_fmax(preds, labels):
    preds = np.round(preds, 2)
    labels = labels.astype(np.int32)
    f_max = 0
    p_max = 0
    r_max = 0
    a_max = 0
    t_max = 0
    for t in range(1, 100):
        threshold = t / 100.0
        predictions = (preds > threshold).astype(np.int32)
        p0 = (preds < threshold).astype(np.int32)
        tp = np.sum(predictions * labels)
        fp = np.sum(predictions) - tp
        fn = np.sum(labels) - tp
        tn = np.sum(p0) - fn
        sn = tp / (1.0 * np.sum(labels))
        sp = np.sum((predictions ^ 1) * (labels ^ 1))
        sp /= 1.0 * np.sum(labels ^ 1)
        fpr = 1 - sp
        precision = tp / (1.0 * (tp + fp))
        recall = tp / (1.0 * (tp + fn))
        f = 2 * precision * recall / (precision + recall)
        acc = (tp + tn) / (tp + fp + tn + fn)
        if p_max < precision:
            f_max = f
            a_max = acc
            p_max = precision
            r_max = recall
            sp_max = sp
            t_max = threshold
    return f_max, a_max, p_max, r_max, t_max

def calculate_f1_score(preds, labels):
    preds = preds.round()
    preds = preds.ravel()
    labels = labels.astype(np.int32)
    labels = labels.ravel()
    acc = accuracy_score(labels, preds)
    precision = precision_score(labels, preds)
    recall = recall_score(labels, preds)
    f = f1_score(labels, preds)
    return f, acc, precision, recall

def evaluate_performance(y_test, y_score):
    n_classes = y_test.shape[1]
    perf = dict()

    perf["M-aupr"] = 0.0
    n = 0
    aupr_list = []
    num_pos_list = []
    for i in range(n_classes):
        num_pos = sum(y_test[:, i])
        num_pos = num_pos.astype(float)
        if num_pos > 0:
            ap = average_precision_score(y_test[:, i], y_score[:, i])
            n += 1
            perf["M-aupr"] += ap
            aupr_list.append(ap)
            num_pos_list.append(num_pos)
    perf["M-aupr"] /= n
    perf['aupr_list'] = aupr_list
    perf['num_pos_list'] = num_pos_list

    # Compute micro-averaged AUPR
    perf['m-aupr'] = average_precision_score(y_test.ravel(), y_score.ravel())
    perf['roc_auc'] = roc_auc_score(y_test.ravel(), y_score.ravel())
    perf['F-max'], perf['acc_max'], perf['pre_max_max'], perf['rec_max'], perf['thr_max'] = calculate_fmax(y_score, y_test)  #precision_max
    perf['F1-score'], perf['accuracy'], perf['precision'], perf['recall'] = calculate_f1_score(y_score, y_test)
    return perf

def evaluate_performance_per(y_test, y_score, index):
    import csv
    out_f = open("perf_per.csv", 'w', newline='')
    writer = csv.writer(out_f)

    index = index[:,0]
    ls_dub = list(index)
    y_score = np.round(y_score, 2)
    y_test = y_test.astype(np.int32)
    threshold = 0.41
    predictions = (y_score > threshold).astype(np.int32)

    for i in range(max(index)+1):
        if i in index:
            ls = [j for j, x in enumerate(ls_dub) if x == i]
            acc = accuracy_score(y_test[ls], predictions[ls])
            num = len(ls)

            writer.writerow([i,num,acc])

    out_f.close()


def get_label_frequency(loc):
    col_sums = loc.sum(0)
    index_lower_300 = np.where(col_sums <= 300)[0]
    index_larger_300 = np.where(col_sums >= 301)[0]
    return index_lower_300, index_larger_300

def get_results(loc, Y_test, y_score):
    perf = defaultdict(dict)
    index_300, index_301 = get_label_frequency(loc)

    perf['Loc<300'] = evaluate_performance(Y_test[:,index_300], y_score[:,index_300])
    perf['Loc>301'] = evaluate_performance(Y_test[:,index_301], y_score[:,index_301])
    perf['all_Loc'] = evaluate_performance(Y_test, y_score)
    return perf

##################   for dsi   #########################
def generate_data(emb, posEdges, negEdges, args):
    #stack codings of two proteins together
    posNum = posEdges.shape[0]
    negNum = negEdges.shape[0]

    X = np.empty((posNum + negNum, 2*emb.shape[-1]))# for noat
    k = 0

    for x in posEdges:  # 遍历边
        X[k] = np.hstack((emb[x[0]],emb[x[1]]))
        k = k + 1
    for x in negEdges:
        X[k] = np.hstack((emb[x[0]],emb[x[1]]))
        k = k + 1

    Y_pos = np.full((posNum,1),[1])
    Y_neg = np.full((negNum,1),[0])
    Y = np.vstack((Y_pos,Y_neg))

    return X,Y

#############################################################################
def id_map(dsi,args):
    entry_list = pd.read_csv(os.path.join(args.data_path, args.species, "data/processing/feature.csv"))
    id_mapping = dict(zip(list(entry_list.index), list(entry_list['Entry'].values)))
    id_mapping2 = dict(zip(list(entry_list['Entry'].values), list(entry_list['Entry name'].values)))
    id_mapping3 = dict(zip(list(entry_list['Entry'].values), list(entry_list['Gene names'].values)))

    dsi['DUB'] = dsi['DUB'].apply(lambda x: id_mapping[x])
    dsi['substrate'] = dsi['substrate'].apply(lambda x: id_mapping[x])

    dsi['DUB_Entry_name'] = dsi['DUB'].apply(lambda x: id_mapping2[x])
    dsi['substrate_Entry_name'] = dsi['substrate'].apply(lambda x: id_mapping2[x])

    dsi['DUB_Gene_name'] = dsi['DUB'].apply(lambda x: id_mapping3[x])
    dsi['substrate_Gene_name'] = dsi['substrate'].apply(lambda x: id_mapping3[x])
    return dsi

def predict_new_dsi(emb, gsp, ind, args):
    #stack codings of two proteins together

    model = torch.load("model.pkl")
    # print(model)
    all_dsi = np.concatenate((gsp, ind))
    df_all_dsi = pd.DataFrame(all_dsi, columns=['DUB_id', 'substrate_id'])
    known_dsi = pd.DataFrame(data=all_dsi, columns=['DUB', 'substrate'])

    dub = list(set(known_dsi['DUB'].values.tolist()))
    X = np.empty((emb.shape[0], 2 * emb.shape[1]))

    L = np.empty((emb.shape[0], 2))

    dsi = pd.DataFrame(columns=['DUB', 'substrate', 'score'])

    string_ppi = pd.read_csv("../human/data/string/ppi_DUB.txt",sep=',')
    string_ppi = string_ppi[["DUB_id","substrate_id"]]
    string_ppi = pd.concat([string_ppi,df_all_dsi,df_all_dsi]).drop_duplicates(keep=False) #Remove the data set in training and testing
    for x in dub:
        for y in range(emb.shape[0]):
            X[y] = np.hstack((emb[x], emb[y]))
            L[y] = np.hstack((x, y))

        X = torch.from_numpy(X).float()
        prediction = model(X)
        prediction = prediction.data.numpy()
        prediction_all = np.concatenate((L, prediction), axis=1)
        pd_data = pd.DataFrame(data=prediction_all, columns=['DUB', 'substrate', 'score'])
        pd_data = pd_data.sort_values(['score'], ascending=False)

        string_ppi = string_ppi[string_ppi['DUB_id'] == x]
        pd_data = pd_data[(pd_data['substrate'].isin(string_ppi['substrate_id'].values))]  # 去除金标准dsi
        pd_data = pd_data.sort_values(['score'], ascending=False)

        dsi = pd.concat([dsi, pd_data], axis=0)

        X = np.empty((emb.shape[0], 2 * emb.shape[1]))
        L = np.empty((emb.shape[0], 2))

    ######################################################################

    pred_dsi = dsi.sort_values(['score'], ascending=False)

    pred_dsi = id_map(pred_dsi,args)

    return pred_dsi

def plot_roc(Y_label, y_pred,str):
    fpr, tpr, threshold = roc_curve(Y_label, y_pred)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(4, 4))
    plt.plot(fpr, tpr, color='red',lw=2, label='sequence (area = %0.4f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title(str+' ROC curve')
    plt.legend(loc="lower right")
    plt.show()

    return roc_auc