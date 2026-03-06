import numpy as np
import torch
from torch.utils.data import DataLoader
import torch_geometric.loader as geom_loader
from torchmetrics import (
    MetricTracker,
    MetricCollection,
)
from typing import Literal, List, Any, Dict

from fiora.GNN.Trainer import Trainer
from fiora.GNN.Losses import WeightedMSELoss, WeightedMAELoss

"""
GNN Trainer
"""

TQDM_DATA_THRESHOLD = 10000


class SpectralTrainer(Trainer):
    def __init__(
        self,
        data: Any,
        train_val_split: float = 0.8,
        split_by_group: bool = False,
        only_training: bool = False,
        train_keys: List[int] | None = None,
        val_keys: List[int] | None = None,
        y_tag: str = "y",
        metric_dict: Dict = None,
        problem_type: Literal[
            "classification", "regression", "softmax_regression"
        ] = "classification",
        library: Literal["standard", "geometric"] = "geometric",
        num_workers: int = 0,
        seed: int = 42,
        device: str = "cpu",
    ):

        super().__init__(
            data,
            train_val_split,
            split_by_group,
            only_training,
            train_keys,
            val_keys,
            seed,
            num_workers,
            device,
        )
        self.y_tag = y_tag
        self.problem_type = problem_type

        # Initialize torch metrics based on dictionary
        if metric_dict:
            self.metrics = {
                data_split: MetricTracker(
                    MetricCollection({t: M() for t, M in metric_dict.items()}),
                    maximize=False,
                ).to(device)
                for data_split in ["train", "val", "masked_val", "test"]
            }
        else:
            self.metrics = self._get_default_metrics(problem_type)
        self.loader_base = (
            geom_loader.DataLoader if library == "geometric" else DataLoader
        )

    @staticmethod
    def _to_float(value):
        if isinstance(value, torch.Tensor):
            return float(value.detach().cpu().item())
        return float(value)

    @staticmethod
    def _build_progress_iterator(dataloader, enabled=False, desc=""):
        if not enabled:
            return dataloader
        try:
            from tqdm.auto import tqdm

            return tqdm(dataloader, total=len(dataloader), desc=desc, leave=False)
        except Exception:
            return dataloader

    @staticmethod
    def _format_metric(stats):
        if "kl" in stats:
            return "kl", float(stats["kl"].detach().cpu().item())
        if "mse" in stats:
            rmse = torch.sqrt(stats["mse"])
            return "rmse", float(rmse.detach().cpu().item())
        if "mae" in stats:
            return "mae", float(stats["mae"].detach().cpu().item())
        if "acc" in stats:
            return "acc", float(stats["acc"].detach().cpu().item())
        key = next(iter(stats.keys()))
        val = stats[key]
        if isinstance(val, torch.Tensor):
            val = float(val.detach().cpu().item())
        return key, float(val)

    def _training_loop(
        self,
        model,
        dataloader,
        optimizer,
        loss_fn,
        metrics,
        with_weights=False,
        with_RT=False,
        with_CCS=False,
        rt_metric=False,
        show_progress=False,
        progress_desc="Train",
    ):
        training_loss = 0
        metrics.increment()
        num_batches = 0

        iterator = self._build_progress_iterator(
            dataloader, enabled=show_progress, desc=progress_desc
        )
        for _, batch in enumerate(iterator):
            # Feed forward
            model.train()

            y_pred = model(batch, with_RT=with_RT, with_CCS=with_CCS)
            kwargs = {}
            if with_weights:
                kwargs = {"weight": batch["weight_tensor"]}
            if getattr(loss_fn, "requires_segment_ptr", False):
                kwargs["segment_ptr"] = y_pred.get("segment_ptr")

            # Compute loss
            loss = loss_fn(
                y_pred["fragment_probs"], batch[self.y_tag], **kwargs
            )  # with logits
            if not rt_metric:
                metrics(
                    y_pred["fragment_probs"], batch[self.y_tag], **kwargs
                )  # call update

            # Add RT and CCS to loss
            if with_RT:
                if with_weights:
                    kwargs["weight"] = batch["weight"][batch["retention_mask"]]
                loss_rt = loss_fn(
                    y_pred["rt"][batch["retention_mask"]],
                    batch["retention_time"][batch["retention_mask"]],
                    **kwargs,
                )
                loss = loss + loss_rt

            if with_CCS:
                if with_weights:
                    kwargs["weight"] = batch["weight"][batch["ccs_mask"]]
                loss_ccs = loss_fn(
                    y_pred["ccs"][batch["ccs_mask"]],
                    batch["ccs"][batch["ccs_mask"]],
                    **kwargs,
                )
                loss = loss + loss_ccs

            if rt_metric:
                metrics(
                    y_pred["rt"][batch["retention_mask"]],
                    batch["retention_time"][batch["retention_mask"]],
                    **kwargs,
                )  # call update
                metrics(
                    y_pred["ccs"][batch["ccs_mask"]],
                    batch["ccs"][batch["ccs_mask"]],
                    **kwargs,
                )  # call update

            # Backpropagate
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            training_loss += self._to_float(loss)
            num_batches += 1

        # End of training cycle: Evaluation
        stats = metrics.compute()
        training_loss /= max(num_batches, 1)
        return stats, training_loss

    def _validation_loop(
        self,
        model,
        dataloader,
        loss_fn,
        metrics,
        with_weights=False,
        with_RT=False,
        with_CCS=False,
        rt_metric=False,
        mask_name=None,
        show_progress=False,
        progress_desc="Validation",
    ):
        metrics.increment()
        validation_loss = 0
        num_batches = 0
        with torch.no_grad():
            iterator = self._build_progress_iterator(
                dataloader, enabled=show_progress, desc=progress_desc
            )
            for _, batch in enumerate(iterator):
                model.eval()
                y_pred = model(batch, with_RT=with_RT, with_CCS=with_CCS)
                if mask_name:
                    kwargs = {}
                    if with_weights:
                        kwargs = {"weight": batch["weight_tensor"][batch[mask_name]]}
                    metrics.update(
                        y_pred["fragment_probs"][batch[mask_name]],
                        batch[self.y_tag][batch[mask_name]],
                        **kwargs,
                    )
                    if not rt_metric and torch.any(batch[mask_name]):
                        batch_loss = loss_fn(
                            y_pred["fragment_probs"][batch[mask_name]],
                            batch[self.y_tag][batch[mask_name]],
                            **kwargs,
                        )
                        validation_loss += self._to_float(batch_loss)
                        num_batches += 1
                else:
                    kwargs = {}
                    if with_weights:
                        kwargs = {"weight": batch["weight_tensor"]}
                    if getattr(loss_fn, "requires_segment_ptr", False):
                        kwargs["segment_ptr"] = y_pred.get("segment_ptr")
                    if not rt_metric:
                        metrics.update(
                            y_pred["fragment_probs"], batch[self.y_tag], **kwargs
                        )
                        batch_loss = loss_fn(
                            y_pred["fragment_probs"], batch[self.y_tag], **kwargs
                        )
                        validation_loss += self._to_float(batch_loss)
                        num_batches += 1
                    if rt_metric:
                        metrics(
                            y_pred["rt"][batch["retention_mask"]],
                            batch["retention_time"][batch["retention_mask"]],
                            **kwargs,
                        )  # call update
                        metrics(
                            y_pred["ccs"][batch["ccs_mask"]],
                            batch["ccs"][batch["ccs_mask"]],
                            **kwargs,
                        )  # call update

        # End of Validation cycle
        stats = metrics.compute()
        if num_batches > 0:
            validation_loss /= num_batches
        else:
            validation_loss = float("nan")
        return stats, validation_loss

    # Training function
    def train(
        self,
        model,
        optimizer,
        loss_fn,
        scheduler=None,
        batch_size=16,
        epochs=2,
        val_every_n_epochs=1,
        use_validation_mask=False,
        with_RT=True,
        with_CCS=True,
        rt_metric=False,
        mask_name="validation_mask",
        save_path: str | None = None,
        tag="",
    ) -> Dict[str, Any]:

        # Set up checkpoint system and model info
        if save_path is None:
            save_path = f"../../checkpoint_{tag}.best.pt"
        self._init_checkpoint_system(save_path=save_path)
        self._init_history()
        model.model_params["training_label"] = self.y_tag

        # Stage data into dataloader
        training_loader = self.loader_base(
            self.training_data,
            batch_size=batch_size,
            num_workers=self.num_workers,
            shuffle=True,
        )
        if not self.only_training:
            validation_loader = self.loader_base(
                self.validation_data,
                batch_size=batch_size,
                num_workers=self.num_workers,
                shuffle=False,
            )
        using_weighted_loss_func = isinstance(
            loss_fn, (WeightedMSELoss, WeightedMAELoss)
        )
        show_train_progress = len(self.training_data) > TQDM_DATA_THRESHOLD
        show_val_progress = (not self.only_training) and (
            len(self.validation_data) > TQDM_DATA_THRESHOLD
        )

        # Main loop
        for e in range(epochs):
            # Training
            train_stats, train_loss = self._training_loop(
                model,
                training_loader,
                optimizer,
                loss_fn,
                self.metrics["train"],
                with_weights=using_weighted_loss_func,
                with_RT=with_RT,
                with_CCS=with_CCS,
                rt_metric=rt_metric,
                show_progress=show_train_progress,
                progress_desc=f"Train {e + 1}/{epochs}",
            )

            # Validation
            is_val_cycle = not self.only_training and (
                (e + 1) % val_every_n_epochs == 0
            )
            if is_val_cycle:
                val_stats, val_loss = self._validation_loop(
                    model,
                    validation_loader,
                    loss_fn,
                    self.metrics["masked_val"]
                    if use_validation_mask
                    else self.metrics["val"],
                    with_weights=using_weighted_loss_func,
                    with_RT=with_RT,
                    with_CCS=with_CCS,
                    rt_metric=rt_metric,
                    mask_name=mask_name if use_validation_mask else None,
                    show_progress=show_val_progress,
                    progress_desc=f"Val {e + 1}/{epochs}",
                )
                val_metric_name, val_metric_value = self._format_metric(val_stats)
            else:
                val_stats, val_loss = None, float("nan")
                val_metric_name, val_metric_value = None, None

            train_metric_name, train_metric_value = self._format_metric(train_stats)
            if val_stats is not None:
                val_metric_str = f"val_{val_metric_name}: {val_metric_value:.4f}"
            else:
                val_metric_str = "val_metric: n/a"
            val_loss_str = f"{val_loss:.4f}" if not np.isnan(val_loss) else "n/a"
            print(
                f"Epoch {e + 1}/{epochs} - loss: {train_loss:.4f} - "
                f"val_loss: {val_loss_str} - "
                f"train_{train_metric_name}: {train_metric_value:.4f} - "
                f"{val_metric_str}"
            )

            # End of epoch: Advance scheduler
            if scheduler:
                if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    last_lr = scheduler.get_last_lr()[0]
                    if is_val_cycle:
                        scheduler.step(val_metric_value)
                        if scheduler.get_last_lr()[0] < last_lr:
                            print(
                                f"\t >> Learning rate reduced from {last_lr:1.0e} to {scheduler.get_last_lr()[0]:1.0e}"
                            )
                else:
                    scheduler.step()

            # Save history
            if is_val_cycle:
                # Update checkpoint
                if val_metric_value < self.checkpoint_stats["val_loss"]:
                    checkpoint_data = {
                        "epoch": e + 1,
                        "val_loss": val_metric_value,
                        "val_metric_name": val_metric_name,
                        "sqrt_val_loss": val_metric_value,
                    }
                    if "mse" in val_stats:
                        checkpoint_data["sqrt_val_loss"] = self._to_float(
                            torch.sqrt(val_stats["mse"])
                        )
                    self._update_checkpoint(
                        checkpoint_data,
                        model,
                    )
                    print(f"\t >> Set new checkpoint to epoch {e + 1}")
                current_lr = (
                    scheduler.get_last_lr()[0]
                    if scheduler is not None
                    else optimizer.param_groups[0]["lr"]
                )
                self._update_history(e + 1, train_stats, val_stats, lr=current_lr)

        print("Finished Training!")
        return self.checkpoint_stats
