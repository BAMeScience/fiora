import torch
from typing import Dict
import torch_geometric.nn as geom_nn
from typing import Literal

from fiora.MOL.constants import ORDERED_ELEMENT_LIST_WITH_HYDROGEN

class EdgePropertyPredictor(torch.nn.Module):
    def __init__(self, edge_feature_dict: Dict, hidden_features: int, static_features: int, out_dimension: int, dense_depth: int=0, dense_dim: int=None, embedding_dim: int=200, embedding_aggregation_type: str='concat', residual_connections: bool=False, subgraph_features: bool=False, pooling_func: Literal["avg", "max"]="avg", input_dropout: float=0, latent_dropout: float=0) -> None:
        ''' Initialize the EdgePropertyPredictor model.
            Args:
                edge_feature_dict (dict): Dictionary containing edge feature information.
                hidden_features (int): Number of hidden features for each layer.
                static_features (int): Number of static features to be concatenated.
                out_dimension (int): Output dimension of the model.
                dense_depth (int, optional): Number of dense layers. Defaults to 0.
                dense_dim (int, optional): Dimension of the dense layers. If None, it will be set to the number of input features. Defaults to None.
                embedding_dim (int, optional): Dimension of the edge embeddings. Defaults to 200.
                embedding_aggregation_type (str, optional): Type of aggregation for edge embeddings. Defaults to 'concat'.
                residual_connections (bool, optional): Whether to use residual connections. Defaults to False.
                input_dropout (float, optional): Dropout rate for input features. Defaults to 0.
                latent_dropout (float, optional): Dropout rate for latent features. Defaults to 0.
        '''
        super().__init__()

        self.activation = torch.nn.ELU()
        #self.edge_embedding = FeatureEmbedding(edge_feature_dict, embedding_dim, aggregation_type=embedding_aggregation_type)
        self.input_dropout = torch.nn.Dropout(input_dropout)
        self.latent_dropout = torch.nn.Dropout(latent_dropout)
        self.residual_connections = residual_connections
        self.subgraph_features = subgraph_features
        self.pooling_func = geom_nn.global_mean_pool if pooling_func == "avg" else geom_nn.global_max_pool
        num_subgraph_features = 2*len(ORDERED_ELEMENT_LIST_WITH_HYDROGEN) + hidden_features*2 if subgraph_features else 0

        dense_layers = []
        num_features = hidden_features*2 + num_subgraph_features + embedding_dim + static_features
        hidden_dimension = dense_dim if dense_dim is not None else num_features
        if hidden_dimension != num_features and residual_connections:
            raise NotImplementedError("Residual connections require the hidden dimension to match the input dimension.")
        for _ in range(dense_depth):
            dense_layers += [torch.nn.Linear(num_features, hidden_dimension)]
            num_features = hidden_dimension
        self.dense_layers = torch.nn.ModuleList(dense_layers)
        
        self.output_layer = torch.nn.Linear(num_features, out_dimension)

    def concat_node_pairs(self, X, batch):
        """
        Concatenates node pairs and optionally adds subgraph features, including element composition.
        This version includes debug messages to trace tensor shapes.
        """
        
        src, dst = batch["edge_index"]
        
        # 1. Node Pair Concatenation
        X_src = X[src]
        X_dst = X[dst]

        node_pairs = torch.cat([X_src, X_dst], dim=1)

        if self.subgraph_features:            
            # 2. Subgraph Feature Pooling
            num_edges = batch.edge_index.size(1)
            edge_batch_map = batch.batch[batch.edge_index[0]]

            left_indices = batch.subgraph_idx_left + batch.ptr[edge_batch_map].unsqueeze(1)
            right_indices = batch.subgraph_idx_right + batch.ptr[edge_batch_map].unsqueeze(1)

            left_batch_vec = torch.arange(num_edges, device=X.device).repeat_interleave(left_indices.size(1))
            right_batch_vec = torch.arange(num_edges, device=X.device).repeat_interleave(right_indices.size(1))

            left_flat = left_indices.flatten()
            right_flat = right_indices.flatten()
            left_mask = left_flat != -1
            right_mask = right_flat != -1

            pooled_left = torch.zeros(num_edges, X.size(1), device=X.device)
            pooled_right = torch.zeros(num_edges, X.size(1), device=X.device)

            if left_mask.any():
                pooled_left = self.pooling_func(X[left_flat[left_mask]], left_batch_vec[left_mask], size=num_edges)
            if right_mask.any():
                pooled_right = self.pooling_func(X[right_flat[right_mask]], right_batch_vec[right_mask], size=num_edges)
            
            subgraph_features = torch.cat([pooled_left, pooled_right], dim=1)
            # 3. Final Concatenation with Subgraph Features
            edge_elem_comp = batch["edge_elem_comp"]
            
            # This is the likely point of failure
            node_pairs = torch.cat([node_pairs, subgraph_features, edge_elem_comp], dim=1)

        return node_pairs
            
    def forward(self, X, batch):  
        # Melt node features into a stack of edges (represented by left and right node)
        X = self.concat_node_pairs(X, batch)
        
        # Add edge features and static features 
        edge_features = batch["edge_embedding"] #self.edge_embedding(batch["edge_attr"])
        edge_features = self.input_dropout(edge_features)
        X = torch.cat([X, edge_features, batch["static_edge_features"]], axis=-1) #self.input_dropout(batch["static_edge_features"])
        
        # Apply fully connected layers
        for layer in self.dense_layers:
            X_skip = X
            X = self.activation(layer(X))
            X = self.latent_dropout(X)
            if self.residual_connections:
                X = X + X_skip

        logits = self.output_layer(X)
        return logits