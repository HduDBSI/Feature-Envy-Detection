import argparse
import math
import numpy as np
import torch
import random
from scipy.spatial.distance import pdist, squareform

def get_parser():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--im_class_num', type=int, default=1, help='imbalance class number')
    parser.add_argument('--class_num', type=int, default=2)
    parser.add_argument('--feature_num', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=1000, help='Number of epochs to train.')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--nhid', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--loss_settings', type=str, default='node_edge', choices=['node_edge', 'edge'])
    parser.add_argument('--binarize', default=True, help='whether binarize calling relationship')  
    parser.add_argument('--up_scale', type=float, default=1)
    parser.add_argument('--edge_weight', type=float, default=1e-6, help='lambda')

    return parser


def split_arti(labels, proportion=[0.64, 0.16, 0.20]):
    num_classes = len(set(labels.tolist()))

    train_idx = []
    val_idx = []
    test_idx = []

    for i in range(num_classes):
        c_idx = (labels == i).nonzero()[:, -1].tolist()
        random.shuffle(c_idx)

        train_num = math.floor(len(c_idx) * proportion[0]) 
        val_num = math.floor(len(c_idx) * proportion[1]) 
        
        train_idx = train_idx + c_idx[:train_num]
        val_idx = val_idx + c_idx[train_num:train_num+val_num]
        test_idx = test_idx + c_idx[train_num+val_num:]

    random.shuffle(train_idx)
    random.shuffle(val_idx)
    random.shuffle(test_idx)

    train_idx = torch.LongTensor(train_idx)
    val_idx = torch.LongTensor(val_idx)
    test_idx = torch.LongTensor(test_idx)

    return train_idx, val_idx, test_idx

def print_edges_num(dense_adj, labels):
    c_num = labels.max().item() + 1
    dense_adj = np.array(dense_adj)
    labels = np.array(labels)

    for i in range(c_num):
        for j in range(c_num):
            row_ind = labels == i
            col_ind = labels == j

            edge_num = dense_adj[row_ind].transpose()[col_ind].sum()
            print("edges between class {:d} and class {:d}: {:f}".format(i, j, edge_num))


from sklearn import metrics
def print_metrics(output, labels, isPrint=True):
    preds = output.max(1)[1].type_as(labels).cpu().detach()
    real = labels.cpu().detach()
    
    Acc = metrics.accuracy_score(real, preds)
    F = metrics.f1_score(real, preds, average='binary')
    P = metrics.precision_score(real, preds, average='binary')
    R = metrics.recall_score(real, preds, average='binary')
    def myRound(a):
        return round(a*100, 2)
    if isPrint:
        print(myRound(Acc), myRound(F), myRound(P), myRound(R))
    return myRound(F)


def recon_upsample(embed, labels, idx_train, adj=None, portion=1.0, im_class_num=1):
    c_largest = labels.max().item()
    avg_number = int(idx_train.shape[0] / (c_largest + 1))

    adj_new = None

    for i in range(im_class_num):
        chosen = idx_train[(labels == (c_largest - i))[idx_train]]
        num = int(chosen.shape[0] * portion)
        if portion == 0:
            c_portion = int(avg_number / chosen.shape[0])
            num = chosen.shape[0]
        else:
            c_portion = 1

        for j in range(c_portion):
            chosen = chosen[:num]

            chosen_embed = embed[chosen, :]
            distance = squareform(pdist(chosen_embed.cpu().detach()))
            np.fill_diagonal(distance, distance.max() + 100)

            idx_neighbor = distance.argmin(axis=-1)

            interp_place = random.random()
            new_embed = embed[chosen, :] + (chosen_embed[idx_neighbor, :] - embed[chosen, :]) * interp_place

            new_labels = labels.new(torch.Size((chosen.shape[0], 1))).reshape(-1).fill_(c_largest - i)
            idx_new = np.arange(embed.shape[0], embed.shape[0] + chosen.shape[0])
            idx_train_append = idx_train.new(idx_new)

            embed = torch.cat((embed, new_embed), 0)
            labels = torch.cat((labels, new_labels), 0)
            idx_train = torch.cat((idx_train, idx_train_append), 0)

            if adj is not None:
                if adj_new is None:
                    adj_new = adj.new(torch.clamp_(adj[chosen, :] + adj[idx_neighbor, :], min=0.0, max=1.0))
                else:
                    temp = adj.new(torch.clamp_(adj[chosen, :] + adj[idx_neighbor, :], min=0.0, max=1.0))
                    adj_new = torch.cat((adj_new, temp), 0)

    if adj is not None:
        add_num = adj_new.shape[0]
        new_adj = adj.new(torch.Size((adj.shape[0] + add_num, adj.shape[0] + add_num))).fill_(0.0)
        new_adj[:adj.shape[0], :adj.shape[0]] = adj[:, :]
        new_adj[adj.shape[0]:, :adj.shape[0]] = adj_new[:, :]
        new_adj[:adj.shape[0], adj.shape[0]:] = torch.transpose(adj_new, 0, 1)[:, :]
        new_adj = dense_to_sparse(new_adj)
        return embed, labels, idx_train, new_adj.detach()
    else:
        return embed, labels, idx_train


def adj_mse_loss(adj_rec, adj_tgt):
    edge_num = adj_tgt.nonzero().shape[0]
    total_num = adj_tgt.shape[0] ** 2

    neg_weight = edge_num / (total_num - edge_num)

    weight_matrix = adj_rec.new(adj_tgt.shape).fill_(1.0)
    weight_matrix[adj_tgt == 0] = neg_weight

    loss = torch.sum(weight_matrix * (adj_rec - adj_tgt) ** 2)
    adj_tgt = dense_to_sparse(adj_tgt)
    return loss


# multiply corresponding element
def sparse_dense_mul(d, s): 
    i = s._indices()
    v = s._values()
    dv = d[i[0,:], i[1,:]]  # get values from relevant entries of dense matrix
    return torch.sparse.FloatTensor(i, v * dv, s.size())


# tranform dense matrix to sparse matrix
def dense_to_sparse(dense):
    if dense.layout == torch.sparse_coo:
        return dense
    idx = torch.nonzero(dense).T  # 这里需要转置一下
    value = dense[idx[0], idx[1]]
    shape = dense.shape
    sparse = dense.detach()
    sparse = torch.sparse_coo_tensor(idx, value, shape)
    return sparse


# save model to file
def save_model(model, file):
    torch.save(model, file)


# keep model in memory
def keep_model(encoder, decoder, classifier):
    model = {}
    model['encoder'] = encoder.state_dict()
    model['decoder'] = decoder.state_dict()
    model['classifier'] = classifier.state_dict()
    return model


# load model from file
def load_model(file, encoder, decoder, classifier):
    loaded_content = torch.load(file, map_location=lambda storage, loc:storage)

    encoder.load_state_dict(loaded_content['encoder'])
    decoder.load_state_dict(loaded_content['decoder'])
    classifier.load_state_dict(loaded_content['classifier'])

    print("successfully loaded: " + file)
    return encoder, decoder, classifier