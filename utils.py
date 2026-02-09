import numpy as np
import scipy.sparse as sp
import torch
import scipy.io as sio
from torch_geometric.data import Data
from torch_geometric.utils import from_scipy_sparse_matrix, to_undirected
from torch_geometric.transforms import NormalizeFeatures
import torch.nn.functional as F

def preprocess_features(features):
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    return r_mat_inv.dot(features).todense()

def normalize_adj(adj):
    adj = adj + sp.eye(adj.shape[0])
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()

def load_data(dataset='cora'):
    data = sio.loadmat(f"./data/{dataset}.mat")
    features = data.get('Attributes', data.get('X'))
    adj = data.get('Network', data.get('A'))
    labels = data.get('Label', data.get('gnd'))
    
    adj = sp.csr_matrix(adj)
    features = sp.lil_matrix(features)
    labels = np.squeeze(np.array(labels))
    
    features_dense = preprocess_features(features)
    adj_normalized = normalize_adj(adj)
    adj_dense = torch.from_numpy(adj_normalized.todense()).float()
    
    edge_index, _ = from_scipy_sparse_matrix(adj_normalized)
    edge_index = to_undirected(edge_index)
    x = torch.tensor(features_dense, dtype=torch.float)
    y = torch.tensor(labels, dtype=torch.long)
    
    pyg_data = Data(x=x, edge_index=edge_index, y=y)
    pyg_data = NormalizeFeatures()(pyg_data)
    
    print(f"Dataset Loaded: {pyg_data.num_nodes} nodes")
    return pyg_data, adj_dense, labels

def random_drop_edges(adj, drop_rate=0.1):
    mask = torch.rand_like(adj) > drop_rate
    return adj * mask.float()

def info_nce_loss(emb1, emb2, temperature=0.07):
    emb1, emb2 = F.normalize(emb1, dim=1), F.normalize(emb2, dim=1)
    sim = torch.mm(emb1, emb2.t()) / temperature
    labels = torch.arange(emb1.size(0), device=emb1.device)
    return F.cross_entropy(sim, labels)