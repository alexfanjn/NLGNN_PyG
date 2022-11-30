import torch
import numpy as np
import torch_geometric
import random

from scipy import io
from torch_geometric.datasets import Planetoid
from sklearn.model_selection import train_test_split

from torch_geometric.utils import sort_edge_index, from_scipy_sparse_matrix, to_scipy_sparse_matrix, degree
import scipy.sparse as sp
from torch.backends import cudnn



def load_heter_data(dataset_name):
    DATAPATH = 'data/heterophily_datasets_matlab'
    fulldata = io.loadmat(f'{DATAPATH}/{dataset_name}.mat')


    # fulldata = io.loadmat(r"https://github.com/alexfanjn/FAGCN_PyG/tree/main/data/heterophily_datasets_matlab/film.mat")

    edge_index = fulldata['edge_index']
    node_feat = fulldata['node_feat']
    label = np.array(fulldata['label'], dtype=np.int32).flatten()

    num_features = node_feat.shape[1]
    num_classes = np.max(label) + 1
    edge_index = torch.tensor(edge_index, dtype=torch.long)
    x = torch.tensor(node_feat)
    y = torch.tensor(label, dtype=torch.long)
    edge_index = torch_geometric.utils.to_undirected(edge_index)
    edge_index, _ = torch_geometric.utils.remove_self_loops(edge_index)
    data = torch_geometric.data.Data(x=x, edge_index=edge_index, y=y)


    return data, num_features, num_classes


def load_homo_data(dataset_name):
    dataset = Planetoid(root='/tmp/'+dataset_name, name=dataset_name)
    return dataset


def set_seed(seed):

    np.random.seed(seed)
    random.seed(seed)
    if seed is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        # torch.cuda.manual_seed_all(seed)
    cudnn.benchmark = False  # if benchmark=True, deterministic will be False
    cudnn.deterministic = True
    return seed


def split_nodes(labels, train_ratio, val_ratio, test_ratio, random_state, split_by_label_flag):
    idx = torch.arange(labels.shape[0])
    if split_by_label_flag:
        idx_train, idx_test = train_test_split(idx, random_state=random_state, train_size=train_ratio+val_ratio, test_size=test_ratio, stratify=labels)
    else:
        idx_train, idx_test = train_test_split(idx, random_state=random_state, train_size=train_ratio+val_ratio, test_size=test_ratio)

    if val_ratio:
        labels_train_val = labels[idx_train]
        if split_by_label_flag:
            idx_train, idx_val = train_test_split(idx_train, random_state=random_state, train_size=train_ratio/(train_ratio+val_ratio), stratify=labels_train_val)
        else:
            idx_train, idx_val = train_test_split(idx_train, random_state=random_state, train_size=train_ratio/(train_ratio+val_ratio))
    else:
        idx_val = None

    return idx_train, idx_val, idx_test


def accuracy(logits, labels):
    _, indices = torch.max(logits, dim=1)
    correct = torch.sum(indices == labels)
    return correct.item()*1.0/len(labels)
