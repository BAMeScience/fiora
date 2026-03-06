import random
from typing import Callable

import numpy as np
import torch
import torch_geometric.loader as geom_loader
from lightning_fabric import Fabric
from torchmetrics import MeanSquaredError

from fiora.GNN.Losses import WeightedMAELoss, WeightedMSELoss

TQDM_DATA_THRESHOLD = 10000


def is_weighted_loss(loss_fn) -> bool:
    return isinstance(loss_fn, (WeightedMSELoss, WeightedMAELoss))


def build_loss_kwargs(
    batch,
    y_pred,
    loss_fn,
    with_weights: bool,
    mask: torch.Tensor | None = None,
    include_segment_ptr: bool = True,
):
    kwargs = {}
    if with_weights:
        kwargs["weight"] = (
            batch["weight_tensor"] if mask is None else batch["weight_tensor"][mask]
        )
    if include_segment_ptr and getattr(loss_fn, "requires_segment_ptr", False):
        kwargs["segment_ptr"] = y_pred.get("segment_ptr")
    return kwargs


def add_rt_ccs_loss(
    loss,
    y_pred,
    batch,
    loss_fn,
    with_weights: bool,
    with_rt: bool,
    with_ccs: bool,
):
    if with_rt:
        kwargs_rt = {}
        if with_weights:
            kwargs_rt["weight"] = batch["weight"][batch["retention_mask"]]
        loss = loss + loss_fn(
            y_pred["rt"][batch["retention_mask"]],
            batch["retention_time"][batch["retention_mask"]],
            **kwargs_rt,
        )
    if with_ccs:
        kwargs_ccs = {}
        if with_weights:
            kwargs_ccs["weight"] = batch["weight"][batch["ccs_mask"]]
        loss = loss + loss_fn(
            y_pred["ccs"][batch["ccs_mask"]],
            batch["ccs"][batch["ccs_mask"]],
            **kwargs_ccs,
        )
    return loss


def safe_metric_update(metric, preds, target, kwargs: dict | None = None):
    kwargs = kwargs or {}
    update = getattr(metric, "update", None)
    if callable(update):
        try:
            update(preds, target, **kwargs)
            return
        except TypeError:
            update(preds, target)
            return
    try:
        metric(preds, target, **kwargs)
    except TypeError:
        metric(preds, target)


def metric_label_and_value(metric_or_stats, preferred_key: str | None = None):
    stats = (
        metric_or_stats.compute()
        if hasattr(metric_or_stats, "compute")
        else metric_or_stats
    )

    if isinstance(stats, dict):
        if preferred_key is not None and preferred_key in stats:
            key = preferred_key
        else:
            for candidate in ("kl", "mse", "mae", "acc"):
                if candidate in stats:
                    key = candidate
                    break
            else:
                key = next(iter(stats.keys()))
        value = stats[key]
    else:
        key = preferred_key or "metric"
        value = stats

    label = "rmse" if key == "mse" else key
    if key == "mse":
        value = torch.sqrt(value)
    if isinstance(value, torch.Tensor):
        value = float(value.detach().cpu().item())
    else:
        value = float(value)
    return label, value


def resolve_fabric_runtime(device: str):
    if device.startswith("cuda"):
        if ":" in device:
            return "cuda", [int(device.split(":")[-1])]
        return "cuda", 1
    if device.startswith("mps"):
        return "mps", 1
    return "cpu", 1


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_progress_iterator(dataloader, enabled=False, desc=""):
    if not enabled:
        return dataloader
    try:
        from tqdm.auto import tqdm

        return tqdm(dataloader, total=len(dataloader), desc=desc, leave=False)
    except Exception:
        return dataloader


def unwrap_model(model):
    return model.module if hasattr(model, "module") else model


def move_batch_to_device(batch, device, non_blocking: bool):
    try:
        return batch.to(device, non_blocking=non_blocking)
    except TypeError:
        return batch.to(device)


def run_epoch(
    fabric: Fabric,
    model: torch.nn.Module,
    dataloader,
    loss_fn,
    metric,
    metric_name: str,
    y_tag: str,
    with_weights: bool,
    with_rt: bool,
    with_ccs: bool,
    rt_metric: bool,
    optimizer=None,
    use_validation_mask: bool = False,
    mask_name: str = "validation_mask",
    show_progress: bool = False,
    progress_desc: str = "",
    non_blocking_transfer: bool = False,
):
    is_training = optimizer is not None
    if is_training:
        model.train()
    else:
        model.eval()
    metric.reset()

    loss_total = 0.0
    loss_batches = 0
    iterator = build_progress_iterator(
        dataloader, enabled=show_progress, desc=progress_desc
    )

    for batch in iterator:
        batch = move_batch_to_device(
            batch, fabric.device, non_blocking=non_blocking_transfer
        )
        with torch.set_grad_enabled(is_training):
            y_pred = model(batch, with_RT=with_rt, with_CCS=with_ccs)

            if use_validation_mask:
                mask = batch[mask_name]
                if torch.any(mask):
                    kwargs = build_loss_kwargs(
                        batch=batch,
                        y_pred=y_pred,
                        loss_fn=loss_fn,
                        with_weights=with_weights,
                        mask=mask,
                        include_segment_ptr=False,
                    )
                    loss = loss_fn(
                        y_pred["fragment_probs"][mask],
                        batch[y_tag][mask],
                        **kwargs,
                    )
                    if not rt_metric:
                        safe_metric_update(
                            metric,
                            y_pred["fragment_probs"][mask],
                            batch[y_tag][mask],
                            kwargs,
                        )
                    else:
                        if with_rt:
                            safe_metric_update(
                                metric,
                                y_pred["rt"][batch["retention_mask"]],
                                batch["retention_time"][batch["retention_mask"]],
                                {},
                            )
                        if with_ccs:
                            safe_metric_update(
                                metric,
                                y_pred["ccs"][batch["ccs_mask"]],
                                batch["ccs"][batch["ccs_mask"]],
                                {},
                            )
                    loss = add_rt_ccs_loss(
                        loss=loss,
                        y_pred=y_pred,
                        batch=batch,
                        loss_fn=loss_fn,
                        with_weights=with_weights,
                        with_rt=with_rt,
                        with_ccs=with_ccs,
                    )
                    loss_total += float(loss.detach().cpu().item())
                    loss_batches += 1
                continue

            kwargs = build_loss_kwargs(
                batch=batch,
                y_pred=y_pred,
                loss_fn=loss_fn,
                with_weights=with_weights,
                include_segment_ptr=True,
            )
            loss = loss_fn(y_pred["fragment_probs"], batch[y_tag], **kwargs)
            if not rt_metric:
                safe_metric_update(
                    metric, y_pred["fragment_probs"], batch[y_tag], kwargs
                )
            else:
                if with_rt:
                    safe_metric_update(
                        metric,
                        y_pred["rt"][batch["retention_mask"]],
                        batch["retention_time"][batch["retention_mask"]],
                        {},
                    )
                if with_ccs:
                    safe_metric_update(
                        metric,
                        y_pred["ccs"][batch["ccs_mask"]],
                        batch["ccs"][batch["ccs_mask"]],
                        {},
                    )

            loss = add_rt_ccs_loss(
                loss=loss,
                y_pred=y_pred,
                batch=batch,
                loss_fn=loss_fn,
                with_weights=with_weights,
                with_rt=with_rt,
                with_ccs=with_ccs,
            )
            loss_total += float(loss.detach().cpu().item())
            loss_batches += 1

            if is_training:
                optimizer.zero_grad(set_to_none=True)
                fabric.backward(loss)
                optimizer.step()

    avg_loss = loss_total / max(loss_batches, 1) if loss_batches > 0 else float("nan")
    metric_label, metric_value = metric_label_and_value(
        metric, preferred_key=metric_name
    )
    return avg_loss, metric_label, metric_value


def train_fabric_loop(
    *,
    model,
    train_data,
    val_data,
    loss_fn,
    metric_dict,
    y_label: str,
    device: str,
    batch_size: int,
    num_workers: int,
    epochs: int,
    val_every: int,
    learning_rate: float,
    weight_decay: float,
    scheduler_name: str,
    scheduler_patience: int,
    scheduler_factor: float,
    with_rt: bool,
    with_ccs: bool,
    rt_metric: bool,
    use_validation_mask: bool,
    validation_mask_name: str,
    output_path: str | None = None,
    optimizer=None,
    scheduler=None,
    progress_threshold: int = TQDM_DATA_THRESHOLD,
    launch_fabric: bool = True,
    logger: Callable[[str], None] | None = print,
    pin_memory: bool | None = None,
):
    has_validation = len(val_data) > 0
    accelerator, devices = resolve_fabric_runtime(device)
    if pin_memory is None:
        pin_memory = accelerator == "cuda"
    use_non_blocking_transfer = bool(pin_memory and accelerator == "cuda")

    fabric = Fabric(accelerator=accelerator, devices=devices)
    if launch_fabric:
        fabric.launch()

    with_weights = is_weighted_loss(loss_fn)
    if metric_dict:
        metric_name, metric_cls = next(iter(metric_dict.items()))
        train_metric = metric_cls().to(fabric.device)
        val_metric = metric_cls().to(fabric.device)
    else:
        metric_name = "mse"
        train_metric = MeanSquaredError().to(fabric.device)
        val_metric = MeanSquaredError().to(fabric.device)

    if optimizer is None:
        optimizer = torch.optim.Adam(
            model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )
    if scheduler is None and scheduler_name == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            patience=scheduler_patience,
            factor=scheduler_factor,
            mode="min",
        )

    train_loader = geom_loader.DataLoader(
        train_data,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True,
        pin_memory=pin_memory,
    )
    val_loader = None
    if has_validation:
        val_loader = geom_loader.DataLoader(
            val_data,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=False,
            pin_memory=pin_memory,
        )

    model, optimizer = fabric.setup(model, optimizer)
    if val_loader is not None:
        train_loader, val_loader = fabric.setup_dataloaders(train_loader, val_loader)
    else:
        train_loader = fabric.setup_dataloaders(train_loader)

    show_train_progress = len(train_data) > progress_threshold
    show_val_progress = has_validation and (len(val_data) > progress_threshold)

    best_metric = float("inf")
    best_epoch = -1
    history = {
        "epoch": [],
        "train_error": [],
        "sqrt_train_error": [],
        "val_error": [],
        "sqrt_val_error": [],
        "lr": [],
    }

    for epoch in range(1, epochs + 1):
        train_loss, train_metric_label, train_metric_value = run_epoch(
            fabric=fabric,
            model=model,
            dataloader=train_loader,
            loss_fn=loss_fn,
            metric=train_metric,
            metric_name=metric_name,
            y_tag=y_label,
            with_weights=with_weights,
            with_rt=with_rt,
            with_ccs=with_ccs,
            rt_metric=rt_metric,
            optimizer=optimizer,
            show_progress=show_train_progress,
            progress_desc=f"Train {epoch}/{epochs}",
            non_blocking_transfer=use_non_blocking_transfer,
        )

        is_val_cycle = has_validation and (epoch % val_every == 0)
        if is_val_cycle:
            val_loss, val_metric_label, val_metric_value = run_epoch(
                fabric=fabric,
                model=model,
                dataloader=val_loader,
                loss_fn=loss_fn,
                metric=val_metric,
                metric_name=metric_name,
                y_tag=y_label,
                with_weights=with_weights,
                with_rt=with_rt,
                with_ccs=with_ccs,
                rt_metric=rt_metric,
                use_validation_mask=use_validation_mask,
                mask_name=validation_mask_name,
                show_progress=show_val_progress,
                progress_desc=f"Val {epoch}/{epochs}",
                non_blocking_transfer=use_non_blocking_transfer,
            )
        else:
            val_loss = float("nan")
            val_metric_label = train_metric_label
            val_metric_value = float("nan")

        monitor_metric = None
        if is_val_cycle:
            monitor_metric = val_metric_value
        elif not has_validation:
            monitor_metric = train_metric_value

        if scheduler is not None:
            prev_lr = optimizer.param_groups[0]["lr"]
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                if monitor_metric is not None and not np.isnan(monitor_metric):
                    scheduler.step(monitor_metric)
            else:
                scheduler.step()
            curr_lr = optimizer.param_groups[0]["lr"]
            if logger is not None and fabric.is_global_zero and curr_lr < prev_lr:
                logger(
                    f"\t >> Learning rate reduced from {prev_lr:1.0e} to {curr_lr:1.0e}"
                )

        if monitor_metric is not None and not np.isnan(monitor_metric):
            if monitor_metric < best_metric:
                best_metric = monitor_metric
                best_epoch = epoch
                if output_path is not None and fabric.is_global_zero:
                    unwrap_model(model).save(output_path)
                    if logger is not None:
                        logger(f"\t >> Set new checkpoint to epoch {epoch}")

        if (is_val_cycle or not has_validation) and fabric.is_global_zero:
            history["epoch"].append(epoch)
            history["train_error"].append(train_metric_value)
            history["sqrt_train_error"].append(train_metric_value)
            history["val_error"].append(
                val_metric_value if is_val_cycle else float("nan")
            )
            history["sqrt_val_error"].append(
                val_metric_value if is_val_cycle else float("nan")
            )
            history["lr"].append(optimizer.param_groups[0]["lr"])

        if logger is not None and fabric.is_global_zero:
            val_loss_str = f"{val_loss:.4f}" if not np.isnan(val_loss) else "n/a"
            val_metric_str = (
                f"{val_metric_value:.4f}" if not np.isnan(val_metric_value) else "n/a"
            )
            logger(
                f"Epoch {epoch}/{epochs} - "
                f"loss: {train_loss:.4f} - "
                f"val_loss: {val_loss_str} - "
                f"train_{train_metric_label}: {train_metric_value:.4f} - "
                f"val_{val_metric_label}: {val_metric_str}"
            )

    if best_epoch < 0:
        best_epoch = epochs
        best_metric = float("nan")
        if output_path is not None and fabric.is_global_zero:
            unwrap_model(model).save(output_path)

    checkpoints = {
        "epoch": best_epoch,
        "val_loss": best_metric,
        "sqrt_val_loss": best_metric,
        "file": output_path,
    }
    return checkpoints, history
