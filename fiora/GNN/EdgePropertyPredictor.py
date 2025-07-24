import torch
from typing import Dict
import torch_geometric.nn as geom_nn


from fiora.MOL.constants import ORDERED_ELEMENT_LIST_WITH_HYDROGEN

class EdgePropertyPredictor(torch.nn.Module):
    def __init__(self, edge_feature_dict: Dict, hidden_features: int, static_features: int, out_dimension: int, dense_depth: int=0, dense_dim: int=None, embedding_dim: int=200, embedding_aggregation_type: str='concat', residual_connections: bool=False, subgraph_features: bool=False, input_dropout: float=0, latent_dropout: float=0) -> None:
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
        self.pooling_func = geom_nn.global_mean_pool
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

        Args:
            X (torch.Tensor): Node features of shape [num_nodes, num_features].
            batch (DataBatch): Batch containing edge_index, subgraph_node_indices, and batch.

        Returns:
            torch.Tensor: Concatenated node pairs with optional subgraph features and element composition.
        """
        src, dst = batch["edge_index"]
        node_pairs = torch.cat([X[src], X[dst]], dim=1)  # Concatenate source and destination node features
        if self.subgraph_features:
            subgraph_features = []
            edge_count = batch["edge_index"].size(1)  # Number of edges
            subgraph_feature_dim = X.size(1)  # Feature dimension of node features

            for graph_id, graph_subgraphs in enumerate(batch["subgraph_node_indices"]):
                for edge_id, subgraph in enumerate(graph_subgraphs):
                    # Adjust local indices to global indices using batch["batch"]
                    left_nodes = subgraph["left"]
                    right_nodes = subgraph["right"]

                    # Convert local indices to global indices
                    left_global_indices = left_nodes + batch.ptr[graph_id]
                    right_global_indices = right_nodes + batch.ptr[graph_id]

                    # Mean-pool node features for left and right subgraphs
                    if len(left_global_indices) > 0:
                        left_features = self.pooling_func(X[left_global_indices], torch.zeros(len(left_global_indices), dtype=torch.int64, device=X.device))
                    else:
                        left_features = torch.zeros(subgraph_feature_dim, device=X.device)

                    if len(right_global_indices) > 0:
                        right_features = self.pooling_func(X[right_global_indices], torch.zeros(len(right_global_indices), dtype=torch.int64, device=X.device))
                    else:
                        right_features = torch.zeros(subgraph_feature_dim, device=X.device)

                    # Ensure left_features and right_features are 1D tensors
                    if left_features.dim() > 1:
                        left_features = left_features.squeeze()
                    if right_features.dim() > 1:
                        right_features = right_features.squeeze()

                    # Concatenate left and right features into a single 1D tensor
                    combined_features = torch.cat([left_features, right_features], dim=0)
                    subgraph_features.append(combined_features)

            # Ensure all tensors in subgraph_features have the same size
            expected_size = 2 * subgraph_feature_dim
            subgraph_features = [sf if sf.size(0) == expected_size else torch.zeros(expected_size, device=X.device) for sf in subgraph_features]

            # Stack subgraph features into a single tensor
            subgraph_features = torch.stack(subgraph_features, dim=0)  # Shape: [num_edges, 2 * subgraph_feature_dim]
            node_pairs = torch.cat([node_pairs, subgraph_features], dim=1)  # Concatenate subgraph features
            node_pairs = torch.cat([node_pairs, batch["edge_elem_comp"]], dim=1)  # Concatenate element composition features

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