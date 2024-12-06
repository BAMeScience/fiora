import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
import torch_geometric.loader as geom_loader
from torchmetrics import Accuracy, MetricTracker, MetricCollection, Precision, Recall, PrecisionRecallCurve, MeanSquaredError, MeanAbsoluteError, R2Score
from sklearn.model_selection import train_test_split
from typing import Literal, List, Callable, Any

from fiora.GNN.Trainer import Trainer
from fiora.GNN.Datasets import collate_graph_batch, collate_graph_edge_batch
from fiora.GNN.Losses import WeightedMSELoss, WeightedMAELoss

'''
GNN Trainer
'''

class SpectralTrainer(Trainer):
    def __init__(self, data, train_val_split: float= 0.8, split_by_group: bool=False, only_training: bool=False, train_keys: List[int]=[], val_keys: List[int]=[], y_tag: str="y", metric_dict=None, problem_type: Literal["classification", "regression", "softmax_regression"]="classification", library: Literal["standard", "geometric"]="geometric", num_workers: int=0, seed: int=42, device: str="cpu"):
        
        super().__init__(data, train_val_split, split_by_group, only_training, train_keys, val_keys, seed, num_workers, device)
        self.y_tag = y_tag
        self.problem_type = problem_type
        
        # Initialize torch metrics based on dictionary 
        if metric_dict:
            self.metrics = {
                data_split: MetricTracker(MetricCollection({
                        t: M() for t,M in metric_dict.items()
                    })).to(device)
                for data_split in ["train", "val", "masked_val", "test"]
            }
        else:
            self.metrics = _get_default_metrics(problem_type)
        self.loader_base = geom_loader.DataLoader if library == "geometric" else DataLoader
        
        # Set output transform according to spectral output format
        self.output_transform = lambda y, batch: y #equivalent to torch.nn.Identity()
        if problem_type == "softmax_regression":
            self.softmax = torch.nn.Softmax(dim=0)
            self.output_transform = self.graph_based_softmax_regression
    
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
        model.model_params["training_label"] = self.y_tag
        training_loader = self.loader_base(self.training_data, batch_size=batch_size, num_workers=self.num_workers, shuffle=True)
        if not self.only_training:
            validation_loader = self.loader_base(self.validation_data, batch_size=batch_size, num_workers=self.num_workers, shuffle=True)
        using_weighted_loss_func = isinstance(loss_fn, WeightedMSELoss) | isinstance(loss_fn, WeightedMAELoss)
        
        # Main loop
        for e in range(epochs):
            self.training_loop(model, training_loader, optimizer, loss_fn, self.metrics["train"], title=f'Epoch {e + 1}/{epochs}: ', with_weights=using_weighted_loss_func, with_RT=with_RT, with_CCS=with_CCS, rt_metric=rt_metric)
            is_val_cycle = not self.only_training and ((e + 1) % val_every_n_epochs == 0)
            if is_val_cycle:
                if masked_validation:
                    val_stats = self.validation_loop(model, validation_loader, loss_fn, self.metrics["masked_val"],  with_weights=using_weighted_loss_func, with_RT=with_RT, with_CCS=with_CCS, mask_name=mask_name, title="Masked Val.", rt_metric=rt_metric)
                else:
                    val_stats = self.validation_loop(model, validation_loader, loss_fn, self.metrics["val"], with_weights=using_weighted_loss_func, with_RT=with_RT, with_CCS=with_CCS, rt_metric=rt_metric)
                if val_stats["mse"].tolist() < checkpoint_stats["val_loss"]:
                    checkpoint_stats["epoch"] = e+1
                    print(f"\t >> Set new checkpoint to epoch {e+1}")
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
    