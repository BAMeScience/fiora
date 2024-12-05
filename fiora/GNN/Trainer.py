from abc import ABC, abstractmethod
import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
import torch_geometric.loader as geom_loader
from torchmetrics import Accuracy, MetricTracker, MetricCollection, Precision, Recall, PrecisionRecallCurve, MeanSquaredError, MeanAbsoluteError, R2Score
from sklearn.model_selection import train_test_split
from typing import Literal, List

from fiora.GNN.Datasets import collate_graph_batch, collate_graph_edge_batch
from fiora.GNN.Losses import WeightedMSELoss, WeightedMAELoss


class Trainer(ABC):
    def __init__(self, data: Dataset, train_val_split: float=0.8, split_by_group: bool=False, only_training: bool=False,
                 train_keys: List[int]=None, val_keys: List[int]=None, seed: int=42, num_workers: int=0, device: str="cpu") -> None:
        
        self.only_training = only_training
        self.num_workers = num_workers
        self.device = device
        
        if only_training:
            self.training_data = data
            self.validation_data = Dataset()
        elif split_by_group:
            self._split_by_group(data, train_val_split, train_keys, val_keys, seed)
        else:
            train_size = int(len(data) * train_val_split)
            self.training_data, self.validation_data = torch.utils.data.random_split(
                data, [train_size, len(data) - train_size], 
                generator=torch.Generator().manual_seed(seed)
                )

    
    def _split_by_group(self, data: Dataset, train_val_split, train_keys, val_keys, seed):
        group_ids = [getattr(x, "group_id") for x in data]
        keys = np.unique(group_ids)
        if train_keys and val_keys:
            self.train_keys, self.val_keys = train_keys, val_keys
            print("Using pre-set train/validation keys")
        else:
            self.train_keys, self.val_keys = train_test_split(
                keys, test_size=1 - train_val_split, random_state=seed
            )
        train_ids = np.where([group_id in self.train_keys for group_id in group_ids])[0]
        val_ids = np.where([group_id in self.val_keys for group_id in group_ids])[0]
        self.training_data = torch.utils.data.Subset(data, train_ids)
        self.validation_data = torch.utils.data.Subset(data, val_ids)

    def _get_default_metrics(self, problem_type: Literal["classification", "regression", "softmax_regression"]):
        metrics = {
            data_split: MetricTracker(MetricCollection( 
                {
                    'acc': Accuracy("binary", num_classes=1), 
                    'prec': Precision('binary', num_classes=1),
                    'rec': Recall('binary', num_classes=1)
                }) if problem_type=="classification" else MetricCollection(
                {
                    'mse': MeanSquaredError(),
                    'mae': MeanAbsoluteError()
                })).to(self.device)
                for data_split in ["train", "val", "masked_val", "test"]
            }
        
        return metrics
    
    
    def is_group_in_training_set(self, group_id):
        return (group_id in self.train_keys)
    
    def is_group_in_validation_set(self, group_id):
        return (group_id in self.val_keys)

    def get_loader(self, dataset, batch_size, shuffle=True):
        return DataLoader(dataset, batch_size=batch_size, num_workers=self.num_workers, shuffle=shuffle)


    @abstractmethod
    def training_loop(self, model, dataloader, optimizer, loss_fn, **kwargs):
        pass

    @abstractmethod
    def validation_loop(self, model, dataloader, loss_fn, **kwargs):
        pass

    @abstractmethod
    def train(self, model, optimizer, loss_fn, **kwargs):
        pass
