import torch
import torch_geometric.nn as geom_nn



class GraphPropertyPredictor(torch.nn.Module):
    def __init__(self, hidden_features: int, static_features: int, out_dimension: int, dense_depth: int=0, dense_dim: int=None, residual_connections: bool=False, input_dropout: float=0, latent_dropout: float=0) -> None:
        ''' Initialize the GraphPropertyPredictor model.
            Args:
                hidden_features (int): Number of hidden features for each layer.
                static_features (int): Number of static features to be concatenated.
                out_dimension (int): Output dimension of the model.
                dense_depth (int, optional): Number of dense layers. Defaults to 0.
                dense_dim (int, optional): Dimension of the dense layers. If None, it will be set to the number of input features. Defaults to None.
                residual_connections (bool, optional): Whether to use residual connections. Defaults to False.
                input_dropout (float, optional): Dropout rate for input features. Defaults to 0.
                latent_dropout (float, optional): Dropout rate for latent features. Defaults to 0.
        '''
        super().__init__()

        self.activation = torch.nn.ELU()
        self.pooling_func = geom_nn.global_mean_pool
        self.input_dropout = torch.nn.Dropout(input_dropout)
        self.latent_dropout = torch.nn.Dropout(latent_dropout)
        self.residual_connections = residual_connections

        dense_layers = []
        num_features = hidden_features + static_features
        hidden_dimension = dense_dim if dense_dim is not None else num_features
        for _ in range(dense_depth):
            dense_layers += [torch.nn.Linear(num_features, hidden_dimension)]
            num_features = hidden_dimension
        self.dense_layers = torch.nn.ModuleList(dense_layers)
        
        self.output_layer = torch.nn.Linear(num_features, out_dimension)
        
    def forward(self, X, batch, covariate_tag="static_graph_features"):
        X = self.pooling_func(X, batch["batch"])
        X = torch.cat([X, batch[covariate_tag]], axis=-1) # self.input_dropout(batch["static_graph_features"])
    
        for layer in self.dense_layers:
            X_skip = X
            X = self.activation(layer(X))
            X = self.latent_dropout(X)
            if self.residual_connections:
                X = X + X_skip
               
        logits = self.output_layer(X)
        
        return logits