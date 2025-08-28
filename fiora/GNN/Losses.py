import torch
from torch import Tensor
from torchmetrics import Metric
from torchmetrics.utilities import dim_zero_cat


###
# Weighted Mean Squared Error    
###
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

###
# Weighted Mean Absolute Error    
###
class WeightedMAELoss(torch.nn.Module):
    def __init__(self):
        super(WeightedMAELoss, self).__init__()

    def forward(self, input, target, weight):
        loss = (weight * torch.abs(input - target))
        return loss.mean()

class WeightedMAEMetric(Metric):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.add_state("sum", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("numel", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: Tensor, target: Tensor, weight: Tensor) -> None:
        self.sum += (weight * torch.abs(preds - target)).sum()
        self.numel += target.numel()

    def compute(self) -> Tensor:
        return self.sum / self.numel

  
class GraphwiseKLLoss(torch.nn.Module):
    """
    KL divergence per-graph over variable-length compiled vectors.
    Expects:
      - y_pred: probabilities (already softmaxed per-graph), shape [N_total]
      - y_true: target nonnegative scores, shape [N_total] (normalized per-graph inside)
      - segment_ptr: LongTensor of shape [num_graphs+1], cumulative boundaries
      - weight (optional): same shape as y_true; reweights targets within each graph
    Reduction:
      - mean: mean over graphs
      - sum: sum over graphs
      - mean_edge: mean over all elements
    """
    requires_segment_ptr = True

    def __init__(self, eps: float = 1e-8, reduction: str = "mean", normalize_targets: bool = True):
        super().__init__()
        self.eps = eps
        self.reduction = reduction
        self.normalize_targets = normalize_targets

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor, segment_ptr: torch.Tensor, weight: torch.Tensor = None):
        assert segment_ptr.dim() == 1 and segment_ptr.numel() >= 2, "segment_ptr must be 1D with at least 2 entries"
        num_graphs = segment_ptr.numel() - 1

        total = y_pred.new_tensor(0.0)
        total_el = 0
        for g in range(num_graphs):
            l = segment_ptr[g].item()
            r = segment_ptr[g+1].item()

            q = y_pred[l:r].clamp_min(self.eps)                  # prob
            p = y_true[l:r].clamp_min(0.0)                       # nonneg
            if weight is not None:
                w = weight[l:r].clamp_min(self.eps)
                p = p * w                                        # weighted targets

            if self.normalize_targets or p.sum() <= 0:
                p = p / p.sum().clamp_min(self.eps)              # normalize per-graph to sum=1

            # KL(p || q) = sum p * (log p - log q)
            kl = (p * (p.clamp_min(self.eps).log() - q.log())).sum()

            total = total + kl
            total_el += (r - l)

        if self.reduction == "sum":
            return total
        elif self.reduction == "mean_edge":
            return total / max(total_el, 1)
        else:  # mean over graphs
            return total / max(num_graphs, 1)
        

class GraphwiseKLLossMetric(Metric):
    """
    Metric counterpart to GraphwiseKLLoss.
    Computes KL(p || q) per graph and reduces by:
      - mean: mean over graphs
      - sum: sum over graphs
      - mean_edge: mean over all elements
    Accepts optional segment_ptr and weight. If segment_ptr is None,
    the whole vector is treated as a single graph.
    """
    def __init__(self, eps: float = 1e-8, reduction: str = "mean", normalize_targets: bool = True, **kwargs):
        super().__init__(**kwargs)
        self.eps = eps
        self.reduction = reduction
        self.normalize_targets = normalize_targets

        self.add_state("total", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total_graphs", default=torch.tensor(0, dtype=torch.long), dist_reduce_fx="sum")
        self.add_state("total_elements", default=torch.tensor(0, dtype=torch.long), dist_reduce_fx="sum")

    def update(self, preds: Tensor, target: Tensor, segment_ptr: Tensor = None, weight: Tensor = None) -> None:
        if segment_ptr is None:
            segment_ptr = torch.tensor([0, preds.numel()], device=preds.device, dtype=torch.long)

        assert segment_ptr.dim() == 1 and segment_ptr.numel() >= 2, "segment_ptr must be 1D with at least 2 entries"
        num_graphs = segment_ptr.numel() - 1

        total = preds.new_tensor(0.0)
        total_el = 0
        for g in range(num_graphs):
            l = segment_ptr[g].item()
            r = segment_ptr[g + 1].item()

            q = preds[l:r].clamp_min(self.eps)         # prob
            p = target[l:r].clamp_min(0.0)             # nonneg
            if weight is not None:
                w = weight[l:r].clamp_min(self.eps)
                p = p * w

            if self.normalize_targets or p.sum() <= 0:
                p = p / p.sum().clamp_min(self.eps)    # normalize per-graph

            kl = (p * (p.clamp_min(self.eps).log() - q.log())).sum()
            total = total + kl
            total_el += (r - l)

        self.total += total
        self.total_graphs += torch.tensor(num_graphs, device=self.total_graphs.device)
        self.total_elements += torch.tensor(total_el, device=self.total_elements.device)

    def compute(self) -> Tensor:
        if self.reduction == "sum":
            return self.total
        elif self.reduction == "mean_edge":
            return self.total / torch.clamp(self.total_elements.float(), min=1.0)
        else:  # "mean"
            return self.total / torch.clamp(self.total_graphs.float(), min=1.0)
