import torch

class MLP(torch.nn.Module):
    def __init__(self, input_features: int, output_dimension: int, hidden_dimension: int, hidden_layers: int = 1):
        super(MLP, self).__init__()
        self.input_features = input_features

        layers = []
        for _ in range(hidden_layers):
            layers += [torch.nn.Linear(input_features, hidden_dimension), torch.nn.ReLU()]
            input_features = hidden_dimension
        self.hidden_layer_sequence = torch.nn.ModuleList(layers)
        
        self.output_layer = torch.nn.Linear(input_features, output_dimension)

    def forward(self, x):
        for layer in self.hidden_layer_sequence:
            x = layer(x)
        logits = self.output_layer(x)
        return logits