import torch
from typing import Dict, Literal
import warnings


class FeatureEmbedding(torch.nn.Module):
    def __init__(self, feature_dict: Dict[str, int], dim=200, aggregation_type=Literal['concat', 'sum']) -> None:
        super().__init__()
        
        self.aggregation_type = aggregation_type
        self.feature_dim = dim
        if aggregation_type == 'concat':
            num_features = len(feature_dict.keys())
            self.feature_dim = int(dim / num_features)
            self.dim = self.feature_dim * num_features
            if self.dim != dim:
                warnings.warn(f"Desired embedding dimension not cleanly dividable by the number of features. Reducing dimension from {dim} to {self.dim}.") 
        elif aggregation_type == 'sum': 
            self.dim = dim
            self.feature_dim = dim
        else:
            raise NameError(f"Unknown aggregation type selected. Valid types are {aggregation_type}.")
        self.embeddings = torch.nn.ModuleList([torch.nn.Embedding(num, self.feature_dim) for feat, num in feature_dict.items()]) # Use ModuleDict instead?
    
    def get_embedding_dimension(self):
        return self.dim
        
    
    def forward(self, features, feature_mask=None):

        node_embeddings = []
        for i, embedding in enumerate(self.embeddings):
            values = features[:, i]
            node_embeddings.append(embedding(values))
        
        if self.aggregation_type == 'sum':
            embedded_features = torch.sum(torch.stack(node_embeddings, dim=-1), dim=-1)
        elif self.aggregation_type == 'concat':
            embedded_features = torch.cat(node_embeddings, dim=-1)
        
        if feature_mask is not None:
            embedded_features = embedded_features * feature_mask.unsqueeze(-1)
        
        return embedded_features




class FeatureEmbeddingPacked(torch.nn.Module):
    def __init__(self, feature_dict: Dict[str, int], dim=200, aggregation_type=Literal['concat', 'sum']) -> None:
        super().__init__()
        
        self.aggregation_type = aggregation_type
        self.feature_dim = dim
        if aggregation_type == 'concat':
            num_features = len(feature_dict.keys())
            self.feature_dim = int(dim / num_features)
            self.dim = self.feature_dim * num_features
            if self.dim != dim:
                warnings.warn(f"Desired embedding dimension not cleanly dividable by the number of features. Reducing dimension from {dim} to {self.dim}.") 
        elif aggregation_type == 'sum': 
            self.dim = dim
            self.feature_dim = dim
        else:
            raise NameError(f"Unknown aggregation type selected. Valid types are {aggregation_type}.")
        self.embeddings = torch.nn.ModuleList([torch.nn.Embedding(num, self.feature_dim) for feat, num in feature_dict.items()]) # Use ModuleDict instead?
    
    def get_embedding_dimension(self):
        return self.dim
        
    
    def forward(self, features, feature_mask=None):

        node_embeddings = []
        for i, embedding in enumerate(self.embeddings):
            values = features[:, :, i]
            node_embeddings.append(embedding(values))
        
        if self.aggregation_type == 'sum':
            embedded_features = torch.sum(torch.stack(node_embeddings, dim=-1), dim=-1)
        elif self.aggregation_type == 'concat':
            embedded_features = torch.cat(node_embeddings, dim=-1)
        
        if feature_mask is not None:
            embedded_features = embedded_features * feature_mask.unsqueeze(-1)
        
        return embedded_features