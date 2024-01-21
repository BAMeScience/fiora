import torch
from torch import Tensor
from torchmetrics import Metric
from torchmetrics.utilities import dim_zero_cat


class WeightedMSELoss(torch.nn.Module):
    def __init__(self):
        super(WeightedMSELoss, self).__init__()

    def forward(self, input, target, weight):
        loss = (weight * (input - target) ** 2)
        return loss.mean()

class WeightedMSEMetric(Metric):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.add_state("sum", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("numel", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: Tensor, target: Tensor, weight: Tensor) -> None:
        self.sum += (weight * (preds - target) ** 2).sum()
        self.numel += target.numel()

    def compute(self) -> Tensor:
        return self.sum / self.numel
        
    
  
  # SUFFERS FROM MEMORY LEAK  
# class WeightedMSEMetric(Metric):
#     def __init__(self, **kwargs):
#         super().__init__(**kwargs)
#         self.add_state("preds", default=[], dist_reduce_fx="cat")
#         self.add_state("target", default=[], dist_reduce_fx="cat")
#         self.add_state("weights", default=[], dist_reduce_fx="cat")

#     def update(self, preds: Tensor, target: Tensor, weight: Tensor) -> None:
#         self.preds.append(preds)
#         self.target.append(target)
#         self.weights.append(weight)

#     def compute(self) -> Tensor:
#         preds = dim_zero_cat(self.preds)
#         target = dim_zero_cat(self.target)
#         weights = dim_zero_cat(self.weights)
#         loss = (weights * (preds - target) ** 2)
#         return loss.mean()