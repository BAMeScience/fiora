import torch
from typing import Dict

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
        subgraph_features = 2*len(ORDERED_ELEMENT_LIST_WITH_HYDROGEN) if subgraph_features else 0

        dense_layers = []
        num_features = hidden_features*2 + subgraph_features + embedding_dim + static_features
        hidden_dimension = dense_dim if dense_dim is not None else num_features
        if hidden_dimension != num_features and residual_connections:
            raise NotImplementedError("Residual connections require the hidden dimension to match the input dimension.")
        for _ in range(dense_depth):
            dense_layers += [torch.nn.Linear(num_features, hidden_dimension)]
            num_features = hidden_dimension
        self.dense_layers = torch.nn.ModuleList(dense_layers)
        
        self.output_layer = torch.nn.Linear(num_features, out_dimension)

    def concat_node_pairs(self, X, batch):
        src, dst = batch["edge_index"]
        node_pairs = torch.cat([X[src], X[dst]], dim=1)
        if self.subgraph_features:
            node_pairs = torch.cat([node_pairs, batch["edge_elem_comp"]], dim=1)  # Add subgraph features
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