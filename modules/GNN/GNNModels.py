import torch
from modules.GNN.GNNLayers import GCNLayer
from modules.GNN.FeatureEmbedding import FeatureEmbedding, FeatureEmbeddingPacked
import torch_geometric.nn as geom_nn


'''
Geometric Models
'''


GeometricLayer = {
    "GraphConv": {
        "Layer": geom_nn.GraphConv,
        "const_args": {'aggr': 'mean'},
        "batch_args": {'edge_index': 'edge_index'}
    },
    "GAT": {
        "Layer": geom_nn.GATConv,
        "const_args": {},
        "batch_args": {'edge_index': 'edge_index'}
    },
    
    "RGCNConv": {
        "Layer": geom_nn.RGCNConv,
        "const_args": {'aggr': 'mean', 'num_relations': 4}, #'num_bases': 30},
        "batch_args": {'edge_index': 'edge_index', 'edge_type': 'edge_type'}
    }
    
}

class GeometricNodeClassifier(torch.nn.Module):
    def __init__(self, in_features, hidden_features, out_dimension, depth, node_feature_dict, embedding_dim=200, embedding_aggregation_type="concat", gnn_type: str="GraphConv") -> None:
        super().__init__()
        self.node_embedding = FeatureEmbedding(node_feature_dict, embedding_dim, aggregation_type=embedding_aggregation_type)
        in_features = self.node_embedding.get_embedding_dimension()
        self.gnn_type = gnn_type
        self.activation = torch.nn.ELU()

        layers = []
        for _ in range(depth):
            layers += [GeometricLayer[gnn_type]["Layer"](in_features, hidden_features, **GeometricLayer[gnn_type]["const_args"])]
            in_features = hidden_features

        self.layers = torch.nn.ModuleList(layers)
        self.output_layer = torch.nn.Linear(in_features, out_dimension)
        
    
    def forward(self, batch):
        X = self.node_embedding(batch["x"])
        
        batch_args = {key: batch[value] for key, value in GeometricLayer[self.gnn_type]["batch_args"].items()}
        for layer in self.layers:
            X = self.activation(layer(X, **batch_args))
        logits = self.output_layer(X)
        
        return logits

class GeometricEdgeClassifier(torch.nn.Module):
    def __init__(self, node_feature_dict, edge_feature_dict, hidden_features: int, static_features: int, out_dimension: int, depth: int, dense_depth: int=0, embedding_dim: int=200, embedding_aggregation_type: str='concat', gnn_type: str="GraphConv", input_dropout: float=0, latent_dropout: float=0) -> None:
        super().__init__()

        self.activation = torch.nn.ELU()
        self.node_embedding = FeatureEmbedding(node_feature_dict, embedding_dim, aggregation_type=embedding_aggregation_type)
        self.edge_embedding = FeatureEmbedding(edge_feature_dict, embedding_dim, aggregation_type=embedding_aggregation_type)
        self.input_dropout = torch.nn.Dropout(input_dropout)
        self.latent_dropout = torch.nn.Dropout(latent_dropout)
        self.gnn_type = gnn_type
        node_features = self.node_embedding.get_embedding_dimension() 
        
        
        
        layers = []
        for _ in range(depth):
            layers += [GeometricLayer[gnn_type]["Layer"](node_features, hidden_features, **GeometricLayer[gnn_type]["const_args"])]
            node_features = hidden_features

        self.graph_layers = torch.nn.ModuleList(layers)
        
        dense_layers = []
        num_features = node_features*2 + self.edge_embedding.get_embedding_dimension() + static_features
        for _ in range(dense_depth):
            dense_layers += [torch.nn.Linear(num_features, num_features)]
        self.dense_layers = torch.nn.ModuleList(dense_layers)
        
        self.output_layer = torch.nn.Linear(num_features, out_dimension)

    def concat_node_pairs(self, X, edge_index):
        src, dst = edge_index
    
        return torch.cat([X[src], X[dst]], dim=1)
        
    def forward(self, batch):

        # Embed node features
        X = self.node_embedding(batch["x"])
        X = self.input_dropout(X)
        
        # Apply graph layers
        batch_args = {key: batch[value] for key, value in GeometricLayer[self.gnn_type]["batch_args"].items()}
        for layer in self.graph_layers:
            X = self.activation(layer(X, **batch_args))
            X = self.latent_dropout(X)
            
        # Melt node features into a stack of edges (represented by left and right node)
        X = self.concat_node_pairs(X, batch["edge_index"])
        
        # Add edge features and static features 
        edge_features = self.edge_embedding(batch["edge_attr"])
        edge_features = self.input_dropout(edge_features)
        X = torch.cat([X, edge_features, batch["static_edge_features"]], axis=-1)
        
        # Apply fully connected layers
        for layer in self.dense_layers:
            X = self.activation(layer(X))
            X = self.latent_dropout(X)
        
        logits = self.output_layer(X)
        return logits

    def graph_embedding(self, batch):
        # Embed node features
        X = self.node_embedding(batch["x"])
        X = self.input_dropout(X)
        
        # Apply graph layers
        batch_args = {key: batch[value] for key, value in GeometricLayer[self.gnn_type]["batch_args"].items()}
        for layer in self.graph_layers:
            X = self.activation(layer(X, **batch_args))
            X = self.latent_dropout(X)
            
        X = geom_nn.global_mean_pool(X, batch["batch"])
        return X



'''
Basic GCN models
'''

class GCNNodeClassifier(torch.nn.Module):
    def __init__(self, in_features, hidden_features, out_dimension, depth, node_feature_dict, embedding_dim=200, embedding_aggregation_type="concat") -> None:
        super().__init__()

        self.node_embedding = FeatureEmbeddingPacked(node_feature_dict, embedding_dim, aggregation_type=embedding_aggregation_type)
        self.activation = torch.nn.ELU()

        layers = []
        for _ in range(depth):
            layers += [GCNLayer(in_features, hidden_features), self.activation]
            in_features = hidden_features

        self.layers = torch.nn.ModuleList(layers)
        self.output_layer = torch.nn.Linear(in_features, out_dimension)
        
    
    def forward(self, X, A):

        for layer in self.layers:
            X = layer(X, A)
        logits = self.output_layer(X)
        
        return logits


class GCNEdgeClassifier(torch.nn.Module):
    def __init__(self, node_feature_dict, edge_feature_dict, hidden_features: int, static_features: int, out_dimension: int, depth: int, dense_depth: int=0, embedding_dim: int=200, embedding_aggregation_type: str='concat', input_dropout: float=0, latent_dropout: float=0) -> None:
        super().__init__()

        self.activation = torch.nn.ELU()
        self.node_embedding = FeatureEmbeddingPacked(node_feature_dict, embedding_dim, aggregation_type=embedding_aggregation_type)
        self.edge_embedding = FeatureEmbeddingPacked(edge_feature_dict, embedding_dim, aggregation_type=embedding_aggregation_type)
        self.input_dropout = torch.nn.Dropout(input_dropout)
        self.latent_dropout = torch.nn.Dropout(latent_dropout)
        node_features = self.node_embedding.get_embedding_dimension() 
        
        
        
        layers = []
        for _ in range(depth):
            layers += [GCNLayer(node_features, hidden_features)]
            node_features = hidden_features

        self.graph_layers = torch.nn.ModuleList(layers)
        
        dense_layers = []
        num_features = node_features*2 + self.edge_embedding.get_embedding_dimension() + static_features
        for _ in range(dense_depth):
            dense_layers += [torch.nn.Linear(num_features, num_features)]
        self.dense_layers = torch.nn.ModuleList(dense_layers)
        
        self.output_layer = torch.nn.Linear(num_features, out_dimension)

    
    '''
    Use helper matrices AL and AR to concatenate hidden features of nodes left and right to an edge. AL and AR must be calculated from adjacency beforehand.  
    '''
    def concat_node_pairs(self, X, AL, AR):
        X = torch.cat([AL@X, AR@X], axis=-1)
        return X
    
    def forward(self, X, A, AL, AR, edge_features, static_features, node_mask, edge_mask):

        # Embed node features
        X = self.node_embedding(X, node_mask)
        X = self.input_dropout(X)
        
        # Apply graph layers
        for layer in self.graph_layers:
            X = layer(X, A)
            X = self.latent_dropout(X)
            
        # Melt node features into a stack of edges (represented by left and right node)
        X = self.concat_node_pairs(X, AL, AR)
        
        # Add edge features and static features 
        edge_features = self.edge_embedding(edge_features, edge_mask)
        edge_features = self.input_dropout(edge_features)
        X = torch.cat([X, edge_features, static_features], axis=-1)
        
        # Apply fully connected layers
        for layer in self.dense_layers:
            X =  self.activation(layer(X))
            X = self.latent_dropout(X)
        
        logits = self.output_layer(X)
        return logits
    
    

class GNNEdgeClassifier(torch.nn.Module):
    def __init__(self, node_feature_dict, edge_feature_dict, hidden_features: int, static_features: int, out_dimension: int, depth: int, dense_depth: int=0, embedding_dim: int=200, embedding_aggregation_type: str='concat', input_dropout: float=0, latent_dropout: float=0) -> None:
        super().__init__()

        self.activation = torch.nn.ELU()
        self.node_embedding = FeatureEmbeddingPacked(node_feature_dict, embedding_dim, aggregation_type=embedding_aggregation_type)
        self.edge_embedding = FeatureEmbeddingPacked(edge_feature_dict, embedding_dim, aggregation_type=embedding_aggregation_type)
        self.input_dropout = torch.nn.Dropout(input_dropout)
        self.latent_dropout = torch.nn.Dropout(latent_dropout)
        node_features = self.node_embedding.get_embedding_dimension() 
        
        
        
        layers = []
        for _ in range(depth):
            layers += [GCNLayer(node_features, hidden_features)]
            node_features = hidden_features

        self.graph_layers = torch.nn.ModuleList(layers)
        
        dense_layers = []
        num_features = node_features*2 + self.edge_embedding.get_embedding_dimension() + static_features
        for _ in range(dense_depth):
            dense_layers += [torch.nn.Linear(num_features, num_features), self.activation]
        self.dense_layers = torch.nn.ModuleList(dense_layers)
        
        self.output_layer = torch.nn.Linear(num_features, out_dimension)

    
    '''
    Use helper matrices AL and AR to concatenate hidden features of nodes left and right to an edge. AL and AR must be calculated from adjacency beforehand.  
    '''
    def concat_node_pairs(self, X, AL, AR):
        X = torch.cat([AL@X, AR@X], axis=-1)
        return X
    
    def forward(self, X, A, AL, AR, edge_features, static_features, node_mask, edge_mask):

        # Embed node features
        X = self.node_embedding(X, node_mask)
        X = self.input_dropout(X)
        
        # Apply graph layers
        for layer in self.graph_layers:
            X = layer(X, A)
            X = self.latent_dropout(X)
            
        # Melt node features into a stack of edges (represented by left and right node)
        X = self.concat_node_pairs(X, AL, AR)
        
        # Add edge features and static features 
        edge_features = self.edge_embedding(edge_features, edge_mask)
        edge_features = self.input_dropout(edge_features)
        X = torch.cat([X, edge_features, static_features], axis=-1)
        
        # Apply fully connected layers
        for layer in self.dense_layers:
            X = layer(X)
            X = self.latent_dropout(X)
        
        logits = self.output_layer(X)
        return logits
    
    
    
    
    
    
    
    
    
    
    #
    # Earlier models
    #
    
    
    class GCNEdgeClassifierOneHot(torch.nn.Module):
        def __init__(self, node_features, hidden_features, additional_features, out_dimension, depth, dense_depth=0) -> None:
            super().__init__()

            self.activation = torch.nn.ELU()
            
            layers = []
            for _ in range(depth):
                layers += [GCNLayer(node_features, hidden_features)]
                node_features = hidden_features

            self.layers = torch.nn.ModuleList(layers)
            
            dense_layers = []
            num_features = node_features*2 + additional_features
            for _ in range(dense_depth):
                dense_layers += [torch.nn.Linear(num_features, num_features), self.activation]
            self.dense_layers = torch.nn.ModuleList(dense_layers)
            
            self.output_layer = torch.nn.Linear(num_features, out_dimension)

        
        '''
        Use helper matrices AL and AR to concatenate hidden features of nodes left and right to an edge. AL and AR must be calculated from adjacency beforehand.  
        '''
        def concat_node_pairs(self, X, AL, AR):
            X = torch.cat([AL@X, AR@X], axis=-1)
            return X
        
        def forward(self, X, A, AL, AR, edge_features):

            #X = self.embedding(X)
            # Apply graph layers
            for layer in self.layers:
                X = layer(X, A)
                
            # Melt node features into edges and concatenate adge features 
            X = self.concat_node_pairs(X, AL, AR)
            X = torch.cat([X, edge_features], axis=-1)
            
            # Apply fully connected layers
            for layer in self.dense_layers:
                X = layer(X)
            logits = self.output_layer(X)
            
            return logits
        