import torch
from torch import Tensor
from torchmetrics import Metric
from torchmetrics.utilities import dim_zero_cat


class WeightedMSE(torch.nn.Module):
    def __init__(self):
        super(WeightedMSE, self).__init__()

    def forward(self, input, target, weight):
        loss = (weight * (input - target) ** 2)
        return loss.mean()

class WeightedMSEMetric(Metric):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.add_state("preds", default=[], dist_reduce_fx="cat")
        self.add_state("target", default=[], dist_reduce_fx="cat")
        self.add_state("weights", default=[], dist_reduce_fx="cat")

    def update(self, preds: Tensor, target: Tensor, weight: Tensor) -> None:
        self.preds.append(preds)
        self.target.append(target)
        self.weights.append(weight)

    def compute(self) -> Tensor:
        preds = dim_zero_cat(self.preds)
        target = dim_zero_cat(self.target)
        weights = dim_zero_cat(self.weights)
        loss = (weights * (preds - target) ** 2)
        return loss.mean()