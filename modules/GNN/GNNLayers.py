import torch

class GCNLayer(torch.nn.Module):
    def __init__(self, in_features, out_features, bias=True) -> None:
        super().__init__()
        self.W1 = torch.nn.Linear(in_features, out_features, bias)
        self.W2 = torch.nn.Linear(in_features, out_features, bias)
        self.activation = torch.nn.ELU()

    def forward(self, X, A):
        HW = self.W1(X)
        AHW = torch.bmm(A, self.W2(X)) #A @ self.W2(X)
        return self.activation(torch.add(HW, AHW))


