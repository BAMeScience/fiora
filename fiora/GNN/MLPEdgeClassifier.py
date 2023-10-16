import torch
import numpy as np

class MLPEdgeClassifier(torch.nn.Module):
    def __init__(self, num_node_features, num_edge_features, num_status_features):
        super(MLPEdgeClassifier, self).__init__()
        self.num_input_features = int(2 * num_node_features + num_edge_features + num_status_features)
        self.size = self.num_input_features
        self.hidden_layer_sequence = torch.nn.Sequential(
            torch.nn.Linear(self.num_input_features, 20, bias=True),
            torch.nn.ReLU(),
        )
        self.output_layer = torch.nn.Sequential(   
            torch.nn.Linear(20,1),
        )

    def forward(self, x, bias=True):
        x = self.hidden_layer_sequence(x)
        logits = self.output_layer(x)
        return logits