
import numpy as np
import matplotlib.pyplot as plt 
import torch

import networkx as nx

node_color_map = {'C': 'gray',
                  'O': 'red',
                  'N': 'blue'}


edge_color_map = {'SINGLE': 'black',
                  'DOUBLE': 'black',
                  'AROMATIC': 'blue'}

edge_width_map = {'SINGLE': 1.5,
                  'DOUBLE': 3,
                  'AROMATIC': 3}




def mol_to_graph(mol):
    G = nx.Graph()

    for atom in mol.GetAtoms():
        color = node_color_map[atom.GetSymbol()] if atom.GetSymbol() in node_color_map.keys() else 'black'
        G.add_node(atom.GetIdx(),
                   atomic_num=atom.GetAtomicNum(),
                   is_aromatic=atom.GetIsAromatic(),
                   atom_symbol=atom.GetSymbol(),
                   color=color,
                   atom=atom)

    for bond in mol.GetBonds():
        G.add_edge(bond.GetBeginAtomIdx(),
                   bond.GetEndAtomIdx(),
                   bond_type=bond.GetBondType(),
                   bond=bond)
    return G

def draw_graph(G, ax=None, edge_labels=False):
    if not ax:
        ax = plt.gca()
    pos = nx.spring_layout(G)
    nx.draw(G,ax=ax, pos=pos,
        labels=nx.get_node_attributes(G, 'atom_symbol'),
        with_labels = True,
        node_color=list(nx.get_node_attributes(G, 'color').values()),
        node_size=800,
        #edges=G.edges(),
        edge_color=[edge_color_map[G[u][v]["bond_type"].name] for u,v in G.edges],
        width=[edge_width_map[G[u][v]["bond_type"].name] for u,v in G.edges],
        )
    if edge_labels:
        nx.draw_networkx_edge_labels(
            G, pos,
            edge_labels=dict([((u, v), f'({u}, {v})') for u, v in G.edges]),
            font_color='red',
            ax=ax
            )

    plt.show()

def get_adjacency_matrix(G):
    #return torch.tensor(nx.convert_matrix.to_numpy_matrix(G), dtype=torch.float32)
    return torch.tensor(nx.convert_matrix.to_numpy_array(G), dtype=torch.float32)

def get_degree_matrix(A):
    #return tf.transpose([tf.clip_by_value(tf.reduce_sum(A, axis=-1), 0.0001, 1000.0)])
    return torch.clamp(torch.sum(A, dim=1, keepdim=True), 0.0001, 1000.0)
    

def get_identity_matrix(A):
    return torch.eye(A.shape[0])

def get_edges(A):
    edge_idx = []

    deg=get_degree_matrix(A)

    row = 0
    for j in range(deg.shape[0]):
        row_degree = int(deg[j,0].numpy())
        for i in range(row_degree):
            edges_to = np.where(A[j] > 0.001)[0]
            edge_idx.append((j, edges_to[i]))
        row += row_degree
    return edge_idx

def compute_edge_related_helper_matrices(A, deg):
    AL = torch.zeros(torch.sum(deg).int(), A.shape[0])
    AR = torch.zeros(AL.shape)
    edge_idx = []

    row = 0
    for j in range(deg.shape[0]):
        row_degree = int(deg[j,0].numpy())
        for i in range(row_degree):
            edges_to = np.where(A[j] > 0.001)[0]
            AL[row + i, j] = 1.0
            AR[row + i, edges_to[i]] = 1.0
            edge_idx.append((j, edges_to[i]))
        row += row_degree

    return AL, AR, edge_idx


def get_helper_matrices_from_edges(edges, A):
    AL = torch.zeros(len(edges), A.shape[0])
    AR = torch.zeros(AL.shape)
    edge_idx = []

    for i, (u,v) in enumerate(edges):
        AL[i, u] = 1.0
        AR[i, v] = 1.0
        
    return AL, AR