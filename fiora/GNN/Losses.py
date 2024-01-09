import torch

class WeightedMSE(torch.nn.Module):
    def __init__(self):
        super(WeightedMSE, self).__init__()

    def forward(self, input, target, weight):
        loss = (weight * (input - target) ** 2)
        return loss.mean()
