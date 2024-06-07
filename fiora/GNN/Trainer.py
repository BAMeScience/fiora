import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
import torch_geometric.loader as geom_loader
from torchmetrics import Accuracy, MetricTracker, MetricCollection, Precision, Recall, PrecisionRecallCurve, MeanSquaredError, MeanAbsoluteError, R2Score
from sklearn.model_selection import train_test_split
from typing import Literal

from fiora.GNN.Datasets import collate_graph_batch, collate_graph_edge_batch
from fiora.GNN.Losses import WeightedMSELoss

'''
GNN Trainer
'''

class Trainer:
    def __init__(self, data, train_val_split: float= 0.8, split_by_group: bool=False, only_training: bool=False, train_keys=None, val_keys=None, y_tag: str="y", metric_dict=None, problem_type: Literal["classification", "regression", "softmax_regression"]="classification", library: Literal["standard", "geometric"]="geometric", num_workers: int=0, seed: int=42, device: str="cpu"):
        
        self.only_training = only_training
        if only_training:
            self.training_data, self.validation_data = data, Dataset()
            group_ids = list((map(lambda x: getattr(x, "group_id"), data)))
            self.train_keys = group_ids
            self.val_keys = []
        
        if split_by_group:
            # Perform train test split that groups data points with the same key together
            group_ids = list((map(lambda x: getattr(x, "group_id"), data)))
            keys = np.unique(group_ids)
            if train_keys is not None and val_keys is not None:
                print("Using pre-arranged train/validation set")
                self.train_keys, self.val_keys = train_keys, val_keys
            else:
                self.train_keys, self.val_keys = train_test_split(keys, test_size=1 - train_val_split, random_state=seed)
            train_ids = np.where(list(map(lambda x: x in self.train_keys, group_ids)))[0]
            val_ids = np.where(list(map(lambda x: x in self.val_keys, group_ids)))[0]
            self.training_data, self.validation_data = torch.utils.data.Subset(data, train_ids), torch.utils.data.Subset(data, val_ids)            
        else:
            train_size = int(len(data)*train_val_split)
            self.training_data, self.validation_data = torch.utils.data.random_split(data, [train_size, len(data) - train_size], generator=torch.Generator().manual_seed(seed))
        
        self.y_tag = y_tag
        self.problem_type = problem_type
        self.num_workers = num_workers
        
        if metric_dict:
            self.metrics = {
                data_split: MetricTracker(MetricCollection({
                        t: M() for t,M in metric_dict.items()
                    })).to(device) # TODO ??????
                for data_split in ["train", "val", "masked_val", "test"]
            }
        else:
            self.metrics = {
                data_split: MetricTracker(MetricCollection({
                        'acc': Accuracy("binary", num_classes=1), 
                        'prec': Precision('binary', num_classes=1),
                        'rec': Recall('binary', num_classes=1)
                    }) if problem_type=="classification" else MetricCollection({
                            'mse': MeanSquaredError(),
                            'mae': MeanAbsoluteError()#,
                            #"r2": R2Score()
                        })
                    ).to(device)
                for data_split in ["train", "val", "masked_val", "test"]
            }
        self.loader_base = DataLoader
        if library=="geometric":
            self.loader_base = geom_loader.DataLoader
        
        self.output_transform = lambda y, batch: y #torch.nn.Identity()
        if problem_type == "softmax_regression":
            self.softmax = torch.nn.Softmax(dim=0)
            self.output_transform = self.graph_based_softmax_regression
    
    def is_group_in_training_set(self, group_id):
        return (group_id in self.train_keys)
    
    def is_group_in_validation_set(self, group_id):
        return (group_id in self.val_keys)
    
    def graph_based_softmax_regression(self, y, batch):      
        edge_graph_map = batch["batch"][batch["edge_index"][0,:]]
        for i in range(batch.num_graphs):
            y[edge_graph_map == i] = self.softmax(y[edge_graph_map == i])
        return 2. * y # times 2, since edges are undirected and therefore doubled
    
    def training_loop(self, model, dataloader, optimizer, loss_fn, metrics, with_weights=False, with_RT=False, with_CCS=False, rt_metric=False, title=""):
        training_loss = 0
        metrics.increment()
                

        for id, batch in enumerate(dataloader):
            
            # Feed forward
            model.train()
            
            y_pred = model(batch, with_RT=with_RT, with_CCS=with_CCS)
            kwargs={}
            if with_weights:
                kwargs={"weight": batch["weight_tensor"]}
             
            loss = loss_fn(y_pred["fragment_probs"], batch[self.y_tag], **kwargs) # with logits
            if not rt_metric: metrics(y_pred["fragment_probs"], batch[self.y_tag], **kwargs) # call update

            # Add RT and CCS to loss
            if with_RT:
                if with_weights:
                    kwargs["weight"] = batch["weight"][batch["retention_mask"]]
                loss_rt = loss_fn(y_pred["rt"][batch["retention_mask"]], batch["retention_time"][batch["retention_mask"]], **kwargs)  
                loss = loss + loss_rt
            
            if with_CCS:
                if with_weights:
                    kwargs["weight"] = batch["weight"][batch["ccs_mask"]]
                loss_ccs = loss_fn(y_pred["ccs"][batch["ccs_mask"]], batch["ccs"][batch["ccs_mask"]], **kwargs)  
                loss = loss + loss_ccs

            if rt_metric:
                metrics(y_pred["rt"][batch["retention_mask"]], batch["retention_time"][batch["retention_mask"]], **kwargs) # call update
                metrics(y_pred["ccs"][batch["ccs_mask"]], batch["ccs"][batch["ccs_mask"]], **kwargs) # call update
                
                    
            # Backpropagate
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            

        # On epoch end: Evaluation
        stats = metrics.compute()
        training_loss /= len(dataloader)
        
        if self.problem_type == "classification":
            print(f'{title} Training Accuracy: {stats["acc"]:>.3f} (Loss per batch: {"NOT TRACKED"})', end='\r')
        else:
            print(f'{title} RMSE: {torch.sqrt(stats["mse"]):>.4f}', end='\r') #MSE: {stats["mse"]:>.3f};  MAE: {stats["mae"]:>.3f}
        return stats
        

    def validation_loop(self, model, dataloader, loss_fn, metrics, with_weights=False, with_RT=False,  with_CCS=False, rt_metric=False,  mask_name=None, title="Validation"):
        metrics.increment()
        with torch.no_grad():
            for id, batch in enumerate(dataloader):
                    model.eval()
                    y_pred = model(batch, with_RT=with_RT, with_CCS=with_CCS)
                    #y_pred = self.output_transform(y_pred, batch) # Transform output in case of softmax_regression

                    if mask_name:
                        loss = loss_fn(y_pred["fragment_probs"][batch[mask_name]], batch[self.y_tag][batch[mask_name]])
                        metrics.update(y_pred["fragment_probs"][batch[mask_name]], batch[self.y_tag][batch[mask_name]])
                    else:
                        kwargs={}
                        if with_weights:
                            kwargs={"weight": batch["weight_tensor"]}
                        loss = loss_fn(y_pred["fragment_probs"], batch[self.y_tag], **kwargs)
                        if not rt_metric:
                            metrics.update(y_pred["fragment_probs"], batch[self.y_tag], **kwargs)
                        if rt_metric:
                            metrics(y_pred["rt"][batch["retention_mask"]], batch["retention_time"][batch["retention_mask"]], **kwargs) # call update
                            metrics(y_pred["ccs"][batch["ccs_mask"]], batch["ccs"][batch["ccs_mask"]], **kwargs) # call update
                        

        # End of Validation cycle
        stats = metrics.compute()
        print(f'\t{title} RMSE: {torch.sqrt(stats["mse"]):>.4f}') #MSE: {stats["mse"]:>.3f}; MAE: {stats["mae"]:>.3f}
        return stats
        
        
    # Training function
    def train(self, model, optimizer, loss_fn, scheduler=None, batch_size=1, epochs=2, val_every_n_epochs=1, masked_validation=False, with_RT=True, with_CCS=True, rt_metric=False, mask_name="validation_mask", tag=""):
        
        checkpoint_stats = {"epoch": -1, "val_loss": 100000.0, "file": f"../../checkpoint_{tag}.best.pt"}
        training_loader = self.loader_base(self.training_data, batch_size=batch_size, num_workers=self.num_workers, shuffle=True)
        if not self.only_training:
            validation_loader = self.loader_base(self.validation_data, batch_size=batch_size, num_workers=self.num_workers, shuffle=True)
        # if isinstance(loss_fn, WeightedMSE):
        #     for data_split in ["train", "val", "masked_val", "test"]:
        #         self.metrics[data_split]["mse"] = WeightedMSE() 
        
        # Main loop
        for e in range(epochs):
            self.training_loop(model, training_loader, optimizer, loss_fn, self.metrics["train"], title=f'Epoch {e + 1}/{epochs}: ', with_weights=isinstance(loss_fn, WeightedMSELoss), with_RT=with_RT, with_CCS=with_CCS, rt_metric=rt_metric)
            is_val_cycle = not self.only_training and ((e + 1) % val_every_n_epochs == 0)
            if is_val_cycle:
                if masked_validation:
                    val_stats = self.validation_loop(model, validation_loader, loss_fn, self.metrics["masked_val"],  with_weights=isinstance(loss_fn, WeightedMSELoss), with_RT=with_RT, with_CCS=with_CCS, mask_name=mask_name, title="Masked Val.", rt_metric=rt_metric)
                else:
                    val_stats = self.validation_loop(model, validation_loader, loss_fn, self.metrics["val"], with_weights=isinstance(loss_fn, WeightedMSELoss), with_RT=with_RT, with_CCS=with_CCS, rt_metric=rt_metric)
                if val_stats["mse"] < checkpoint_stats["val_loss"]:
                    checkpoint_stats["epoch"] = e+1
                    checkpoint_stats["val_loss"] = val_stats["mse"].tolist()
                    model.save(checkpoint_stats["file"])
            if scheduler:
                if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    # step only after validation
                    if is_val_cycle:
                        scheduler.step(torch.sqrt(val_stats["mse"]))
                else:
                    scheduler.step()
            
        print("Finished Training!")
        return checkpoint_stats
    