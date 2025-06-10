import torch


class EdgePropertyPredictor(torch.nn.Module):
    def __init__(self, edge_feature_dict, hidden_features: int, static_features: int, out_dimension: int, dense_depth: int=0, embedding_dim: int=200, embedding_aggregation_type: str='concat', residual_connections: bool=False, input_dropout: float=0, latent_dropout: float=0) -> None:
        super().__init__()

        self.activation = torch.nn.ELU()
        #self.edge_embedding = FeatureEmbedding(edge_feature_dict, embedding_dim, aggregation_type=embedding_aggregation_type)
        self.input_dropout = torch.nn.Dropout(input_dropout)
        self.latent_dropout = torch.nn.Dropout(latent_dropout)
        self.residual_connections = residual_connections

        dense_layers = []
        num_features = hidden_features*2 + embedding_dim + static_features
        for _ in range(dense_depth):
            dense_layers += [torch.nn.Linear(num_features, num_features)]
        self.dense_layers = torch.nn.ModuleList(dense_layers)
        
        self.output_layer = torch.nn.Linear(num_features, out_dimension)

    def concat_node_pairs(self, X, edge_index):
        src, dst = edge_index
    
        return torch.cat([X[src], X[dst]], dim=1)
        
    def forward(self, X, batch):
        
        # Melt node features into a stack of edges (represented by left and right node)
        X = self.concat_node_pairs(X, batch["edge_index"])
        
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