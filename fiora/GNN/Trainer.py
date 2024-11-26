from abc import ABC, abstractmethod
import torch
from torch.utils.data import DataLoader



class Trainer(ABC):
    def __init__(self, data, train_val_split=0.8, only_training=False, split_by_group=False,
                 train_keys=None, val_keys=None, seed=42, num_workers=0, device="cpu"):
        
        self.only_training = only_training
        self.num_workers = num_workers
        self.device = device
        
        if only_training:
            self.training_data = data
            self.validation_data = None
        elif split_by_group:
            self._split_by_group(data, train_val_split, train_keys, val_keys, seed)
        else:
            train_size = int(len(data) * train_val_split)
            self.training_data, self.validation_data = torch.utils.data.random_split(
                data, [train_size, len(data) - train_size], generator=torch.Generator().manual_seed(seed)
            )
    
    def _split_by_group(self, data, train_val_split, train_keys, val_keys, seed):
        group_ids = [getattr(x, "group_id") for x in data]
        keys = np.unique(group_ids)
        if train_keys and val_keys:
            self.train_keys, self.val_keys = train_keys, val_keys
        else:
            self.train_keys, self.val_keys = train_test_split(
                keys, test_size=1 - train_val_split, random_state=seed
            )
        train_ids = np.where([group_id in self.train_keys for group_id in group_ids])[0]
        val_ids = np.where([group_id in self.val_keys for group_id in group_ids])[0]
        self.training_data = torch.utils.data.Subset(data, train_ids)
        self.validation_data = torch.utils.data.Subset(data, val_ids)

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
