import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset

'''
class AtomAromaticityData(Dataset):
    def __init__(self, df) -> None:
        self.X = np.concatenate(df["features"].values, dtype='float32')
        self.y = np.concatenate(df["is_aromatic"].values*1, dtype='float32')
    
    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return [self.X[idx], self.y[idx]]

    def num_features(self):
        return self.X.shape[1]
'''    



class AtomAromaticityData(Dataset):
    def __init__(self, df) -> None:

        self.X = df["features"].apply(lambda x: torch.tensor(x, dtype=torch.float32)).values
        self.A = df["Atilde"].apply(lambda x: torch.tensor(x, dtype=torch.float32)).values
        self.y = df["is_aromatic"].apply(lambda x: torch.tensor(x, dtype=torch.float32)).values*1

    
    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return [self.X[idx], self.A[idx], self.y[idx]]

    def num_features(self):
        return self.X[0].shape[1]

class SimpleNodeData(Dataset):
    def __init__(self, data: pd.Series, feature_tag: str, label: str, device="cpu") -> None:
        self.X = torch.cat(data.apply(lambda x: getattr(x, feature_tag).to(device)).to_list())
        self.y = torch.cat(data.apply(lambda x: getattr(x, label).to(device)).to_list())
    
    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

    def num_features(self):
        return self.X.shape[1]
    
    
class NodeSingleLabelData(Dataset):
    def __init__(self, data: pd.Series, feature_tag: str, adj_tag: str, label: str, device="cpu") -> None:

        self.X = data.apply(lambda x: getattr(x, feature_tag).to(device)).values
        self.A = data.apply(lambda x: getattr(x, adj_tag).to(device)).values
        self.num_nodes = np.array(list(map(lambda x: x.shape[0], self.X)))
        self.y = data.apply(lambda x: getattr(x, label).to(device)).values
        
    
    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return [self.X[idx], self.A[idx], self.num_nodes[idx], self.y[idx]]

    def num_features(self):
        return self.X[0].shape[1]
    

class EdgeSingleLabelData(Dataset):
    def __init__(self, data: pd.Series, feature_tag: str, left_tag: str, right_tag: str, adj_tag: str, edge_feature_tag: str, static_feature_tag: str, label: str, validation_mask_tag: str, group_id: str, device="cpu") -> None:

        self.X = data.apply(lambda x: getattr(x, feature_tag).to(device)).values
        self.A = data.apply(lambda x: getattr(x, adj_tag).to(device)).values
        self.AL = data.apply(lambda x: getattr(x, left_tag).to(device)).values # matrix to list all nodes to the left of an edge
        self.AR = data.apply(lambda x: getattr(x, right_tag).to(device)).values # matrix to list all nodes to the right of an edge
        self.num_nodes = np.array(list(map(lambda x: x.shape[0], self.X)))
        self.y = data.apply(lambda x: getattr(x, label).to(device)).values
        self.num_edges = np.array(list(map(lambda x: x.shape[0], self.y)))
        self.edge_features = data.apply(lambda x: getattr(x, edge_feature_tag).to(device)).values
        self.static_features = data.apply(lambda x: getattr(x, static_feature_tag).to(device)).values
        self.validation_mask = data.apply(lambda x: getattr(x, validation_mask_tag).to(device)).values 
        self.group_id = data.apply(lambda x: getattr(x, group_id)).values
        
    
    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return [self.X[idx], self.A[idx], self.num_nodes[idx], self.num_edges[idx], self.edge_features[idx], self.static_features[idx], self.y[idx], self.AL[idx], self.AR[idx], self.validation_mask[idx]]

    def num_features(self): # Number of node features
        return self.X[0].shape[1]
    
    def num_static_features(self): # Number of additional features concatenated for the edge classification
        return self.static_features[0].shape[1]
    
    def get_unique_groups(self):
        return np.unique(self.group_id)
    
    def get_indices_of_groups(self, groups):
        indices = np.array([], dtype=int)
        for g in groups:
            ids = np.where(self.group_id == g)[0]
            indices = np.concatenate((indices, ids), axis=0)
        return indices

'''
def collate_graph_batch(batch):
    X, A, no_nodes, y = zip(*batch)
    max_nodes = np.max(no_nodes)
    pad_sizes = max_nodes - no_nodes
    X = torch.nn.utils.rnn.pad_sequence(X, batch_first=True, padding_value=0)
    pad_matrix = lambda x, y: torch.nn.functional.pad(x, (0,y,0,y), value=0)
    A = torch.stack(list(map(pad_matrix, A, pad_sizes)))
    y = torch.nn.utils.rnn.pad_sequence(y, batch_first=True, padding_value=0)
    return X, A, y
'''

def collate_graph_batch(batch):
    X, A, num_nodes, y = zip(*batch)
    max_nodes = np.max(num_nodes)
    pad_sizes = max_nodes - num_nodes
    X = torch.nn.utils.rnn.pad_sequence(X, batch_first=True, padding_value=0)
    pad_matrix = lambda x, y: torch.nn.functional.pad(x, (0,y,0,y), value=0)
    A = torch.stack(list(map(pad_matrix, A, pad_sizes)))
    y = torch.nn.utils.rnn.pad_sequence(y, batch_first=True, padding_value=0)
    node_mask = torch.zeros((X.shape[0], max_nodes), dtype=torch.bool).to(X.get_device())
    adj_mask = torch.zeros((X.shape[0], max_nodes, max_nodes), dtype=torch.bool)
    for i in range(node_mask.shape[0]):
        node_mask[i, :num_nodes[i]] = 1
        adj_mask[i, :num_nodes[i], :num_nodes[i]] = 1
        
    batch_record = {
        'X': X,
        'A': A,
        'y': y,
        'node_mask': node_mask,
        'adj_mask': adj_mask,
        'num_of_nodes': torch.tensor(list(num_nodes)).unsqueeze(dim=1)
    }
    return batch_record


def collate_graph_edge_batch(batch):
    X, A, num_nodes, num_edges, edge_features, static_features, y, AL, AR, validation_bits = zip(*batch)
    max_nodes = np.max(num_nodes)
    pad_sizes = max_nodes - num_nodes
    max_edges = np.max(num_edges)
    

    X = torch.nn.utils.rnn.pad_sequence(X, batch_first=True, padding_value=0)
    edge_features = torch.nn.utils.rnn.pad_sequence(edge_features, batch_first=True, padding_value=0)
    static_features = torch.nn.utils.rnn.pad_sequence(static_features, batch_first=True, padding_value=0)
    pad_matrix = lambda x, y: torch.nn.functional.pad(x, (0,y,0,y), value=0)
    A = torch.stack(list(map(pad_matrix, A, pad_sizes)))
    y = torch.nn.utils.rnn.pad_sequence(y, batch_first=True, padding_value=0)
    
    pad_matrix = lambda x, y, z: torch.nn.functional.pad(x, (0,y,0,z), value=0) # Pad helper matrices to maximum size
    pad_edge_sizes = max_edges - num_edges
    AL = torch.stack(list(map(pad_matrix, AL, pad_sizes, pad_edge_sizes)))
    AR = torch.stack(list(map(pad_matrix, AR, pad_sizes, pad_edge_sizes)))
    
    node_mask = torch.zeros((X.shape[0], max_nodes), dtype=torch.bool).to(X.get_device())
    adj_mask = torch.zeros((X.shape[0], max_nodes, max_nodes), dtype=torch.bool)
    
    y_mask = torch.zeros((y.shape[0], max_edges), dtype=torch.bool).to(X.get_device())
    validation_mask = torch.zeros((y.shape[0], max_edges), dtype=torch.bool)

    for i in range(node_mask.shape[0]):
        node_mask[i, :num_nodes[i]] = 1
        adj_mask[i, :num_nodes[i], :num_nodes[i]] = 1
        y_mask[i, :num_edges[i]] = 1
        validation_mask[i, :num_edges[i]] = validation_bits[i].flatten()
    
    batch_record = {
        'X': X,
        'A': A,
        'y': y,
        'AL': AL,
        'AR': AR,
        'node_mask': node_mask,
        'adj_mask': adj_mask,
        'y_mask': y_mask,
        'edge_features': edge_features,
        'static_features': static_features,
        'validation_mask': validation_mask,
        'num_of_nodes': torch.tensor(list(num_nodes)).unsqueeze(dim=1),
        'num_of_edges': torch.tensor(list(num_edges)).unsqueeze(dim=1)
    }
    return batch_record