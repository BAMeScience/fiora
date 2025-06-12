import torch
import torch_geometric.nn as geom_nn
from typing import Literal

'''
Geometric Models
'''

GeometricLayer = {
    "GraphConv": {
        "Layer": geom_nn.GraphConv,
        "divide_output_dim": False,
        "const_args": {'aggr': 'mean'},
        "batch_args": {'edge_index': 'edge_index'}
    },
    "GAT": {
        "Layer": geom_nn.GATConv,
        "divide_output_dim": True,
        "const_args": {'heads': 5},
        "batch_args": {'edge_index': 'edge_index', 'edge_attr': 'edge_embedding'}
    },
    
    "RGCNConv": {
        "Layer": geom_nn.RGCNConv,
        "divide_output_dim": False,
        "const_args": {'aggr': 'mean', 'num_relations': 4},
        "batch_args": {'edge_index': 'edge_index', 'edge_type': 'edge_type'}
    },
    
    "TransformerConv": {
        "Layer": geom_nn.TransformerConv,
        "divide_output_dim": True,
        "const_args": {'heads': 5, 'edge_dim': 300},
        "batch_args": {'edge_index': 'edge_index', 'edge_attr': 'edge_embedding'}
    },
    
    "CGConv": {
        "Layer": geom_nn.CGConv,
        "divide_output_dim": False,
        "const_args": {'aggr': "mean"}, #, 'dim': 300},
        "batch_args": {'edge_index': 'edge_index', 'edge_attr': 'edge_embedding'}
    }
}

'''
Graph Neural Network (GNN) Class
'''

class GNN(torch.nn.Module):
    def __init__(self, hidden_features: int, depth: int, embedding_dim: int=None, embedding_aggregation_type: str='concat', gnn_type: Literal["GraphConv", "GAT", "RGCNConv", "TransformerConv", "CGConv"]="RGCNConv", residual_connections: bool=False, layer_stacking: bool=False, input_dropout: float=0, latent_dropout: float=0) -> None:
        ''' Initialize the GNN model.
        Args:
            hidden_features (int): Number of hidden features for each layer.
            depth (int): Number of graph layers.
            embedding_dim (int, optional): Dimension of the node embeddings. Defaults to None.
            embedding_aggregation_type (str, optional): Type of aggregation for node embeddings. Defaults to 'concat'.
            gnn_type (Literal, optional): Type of GNN layer to use. Defaults to "RGCNConv".
            residual_connections (bool, optional): Whether to use residual connections. Defaults to False.
            input_dropout (float, optional): Dropout rate for input features. Defaults to 0.
            latent_dropout (float, optional): Dropout rate for latent features. Defaults to 0.
        '''

        super().__init__()

        self.activation = torch.nn.ELU()
        self.input_dropout = torch.nn.Dropout(input_dropout)
        self.latent_dropout = torch.nn.Dropout(latent_dropout)
        self.gnn_type = gnn_type
        self.residual_connections = residual_connections
        self.layer_stacking = layer_stacking
        node_features =  embedding_dim
        
        
        layers = []
        self.layer_norms = torch.nn.ModuleList()
        for _ in range(depth):
            layers += [
                GeometricLayer[gnn_type]["Layer"](
                    node_features, 
                    int(hidden_features / GeometricLayer[gnn_type]["const_args"]["heads"]) 
                    if GeometricLayer[gnn_type]["divide_output_dim"] 
                    else hidden_features, **GeometricLayer[gnn_type]["const_args"])]
            self.layer_norms.append(torch.nn.LayerNorm(hidden_features))
            node_features = hidden_features

        self.graph_layers = torch.nn.ModuleList(layers)


    def forward(self, batch):
        # Initialize node embeddings
        X = batch["node_embedding"]
        X = self.input_dropout(X)

        # If layer stacking is enabled, stack the node features
        stacked_embeddings = [X] if self.layer_stacking else []

        # Apply graph layers
        batch_args = {key: batch[value] for key, value in GeometricLayer[self.gnn_type]["batch_args"].items()}
        for i, layer in enumerate(self.graph_layers):
            X_skip = X
            X = layer(X, **batch_args)
            X = self.layer_norms[i](X)
            X = self.activation(X)
            X = self.latent_dropout(X)
            if self.residual_connections:
                X = X + X_skip

            if self.layer_stacking:
                stacked_embeddings.append(X)

        if self.layer_stacking:
            X = torch.cat(stacked_embeddings, dim=-1)
        return X

    def get_embedding_dimension(self):
        """Get the output dimension of the GNN."""
        return self.graph_layers[-1].out_channels * (len(self.graph_layers) + 1 if self.layer_stacking else 1)