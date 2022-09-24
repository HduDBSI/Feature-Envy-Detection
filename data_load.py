import scipy.sparse as sp
import numpy as np
import torch
import pandas as pd

def load_data_custom(project_ID):
    nodes_data = pd.read_csv('data/' + project_ID + '/method.csv')
    edges_data = pd.read_csv('data/' + project_ID + '/method-invocate-method.csv')

    def get_tensor(dataframe_series, isFloat=False):
        if isFloat:
            return torch.tensor(data=dataframe_series.values).float()
        else:
            return torch.tensor(data=dataframe_series.values)
    
    features = get_tensor(
        nodes_data[['LOC','CC','PC','NCM','NCMEC','NCMIC','NECA','NAMFAEC']], 
        isFloat=True)

    labels = get_tensor(nodes_data['label'])
    edges = get_tensor(edges_data[['Src','Dst']])

    # make adjacency matrix
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(labels.shape[0], labels.shape[0]),
                        dtype=np.float32)

    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    # Normalize features
    features = normalize(features)
    features = torch.FloatTensor(features)

    # utils.print_edges_num(adj.todense(), labels)
    adj = sparse_mx_to_torch_sparse_tensor(adj)

    return adj, features, labels

def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)
