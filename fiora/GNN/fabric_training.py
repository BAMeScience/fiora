import random
import warnings
from dataclasses import dataclass
from typing import Callable

import numpy as np
import torch
import torch_geometric.loader as geom_loader
from lightning_fabric import Fabric
from lightning_fabric.utilities.warnings import PossibleUserWarning
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
    weight_tensor_override: torch.Tensor | None = None,
):
    kwargs = {}
    if with_weights:
        weight_tensor = (
            weight_tensor_override
            if weight_tensor_override is not None
            else batch['weight_tensor']
        )
        kwargs['weight'] = weight_tensor if mask is None else weight_tensor[mask]
    if include_segment_ptr and getattr(loss_fn, 'requires_segment_ptr', False):
        kwargs['segment_ptr'] = y_pred.get('segment_ptr')
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
            kwargs_rt['weight'] = batch['weight'][batch['retention_mask']]
        loss = loss + loss_fn(
            y_pred['rt'][batch['retention_mask']],
            batch['retention_time'][batch['retention_mask']],
            **kwargs_rt,
        )
    if with_ccs:
        kwargs_ccs = {}
        if with_weights:
            kwargs_ccs['weight'] = batch['weight'][batch['ccs_mask']]
        loss = loss + loss_fn(
            y_pred['ccs'][batch['ccs_mask']],
            batch['ccs'][batch['ccs_mask']],
            **kwargs_ccs,
        )
    return loss


def safe_metric_update(metric, preds, target, kwargs: dict | None = None):
    kwargs = kwargs or {}
    update = getattr(metric, 'update', None)
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
        if hasattr(metric_or_stats, 'compute')
        else metric_or_stats
    )

    if isinstance(stats, dict):
        if preferred_key is not None and preferred_key in stats:
            key = preferred_key
        else:
            for candidate in ('kl', 'mse', 'mae', 'acc'):
                if candidate in stats:
                    key = candidate
                    break
            else:
                key = next(iter(stats.keys()))
        value = stats[key]
    else:
        key = preferred_key or 'metric'
        value = stats

    label = 'rmse' if key == 'mse' else key
    if key == 'mse':
        value = torch.sqrt(value)
    if isinstance(value, torch.Tensor):
        value = float(value.detach().cpu().item())
    else:
        value = float(value)
    return label, value


def resolve_fabric_runtime(device: str):
    if device.startswith('cuda'):
        if ':' in device:
            return 'cuda', [int(device.split(':')[-1])]
        return 'cuda', 1
    if device.startswith('mps'):
        return 'mps', 1
    return 'cpu', 1


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_progress_iterator(dataloader, enabled=False, desc=''):
    if not enabled:
        return dataloader
    try:
        from tqdm.auto import tqdm

        return tqdm(dataloader, total=len(dataloader), desc=desc, leave=False)
    except Exception:
        return dataloader


def unwrap_model(model):
    return model.module if hasattr(model, 'module') else model


def apply_precursor_loss_weight(
    weight_tensor: torch.Tensor,
    segment_ptr: torch.Tensor | None,
    precursor_loss_weight: float,
) -> torch.Tensor:
    if precursor_loss_weight == 1.0:
        return weight_tensor
    if segment_ptr is None or segment_ptr.numel() < 2:
        return weight_tensor

    weighted = weight_tensor.clone()
    starts = segment_ptr[:-1]
    ends = segment_ptr[1:]
    lengths = ends - starts
    valid = lengths >= 2
    if torch.any(valid):
        right_idx = ends[valid] - 1
        left_idx = ends[valid] - 2
        weighted[right_idx] = weighted[right_idx] * precursor_loss_weight
        weighted[left_idx] = weighted[left_idx] * precursor_loss_weight
    return weighted


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
    mask_name: str = 'validation_mask',
    show_progress: bool = False,
    progress_desc: str = '',
    non_blocking_transfer: bool = False,
    precursor_loss_weight: float = 1.0,
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
            use_weight_vector = with_weights or getattr(
                loss_fn, 'requires_segment_ptr', False
            )
            weight_tensor = None
            if use_weight_vector:
                weight_tensor = apply_precursor_loss_weight(
                    batch['weight_tensor'],
                    y_pred.get('segment_ptr'),
                    precursor_loss_weight,
                )

            if use_validation_mask:
                mask = batch[mask_name]
                if torch.any(mask):
                    kwargs = build_loss_kwargs(
                        batch=batch,
                        y_pred=y_pred,
                        loss_fn=loss_fn,
                        with_weights=use_weight_vector,
                        mask=mask,
                        include_segment_ptr=False,
                        weight_tensor_override=weight_tensor,
                    )
                    loss = loss_fn(
                        y_pred['fragment_probs'][mask],
                        batch[y_tag][mask],
                        **kwargs,
                    )
                    if not rt_metric:
                        safe_metric_update(
                            metric,
                            y_pred['fragment_probs'][mask],
                            batch[y_tag][mask],
                            kwargs,
                        )
                    else:
                        if with_rt:
                            safe_metric_update(
                                metric,
                                y_pred['rt'][batch['retention_mask']],
                                batch['retention_time'][batch['retention_mask']],
                                {},
                            )
                        if with_ccs:
                            safe_metric_update(
                                metric,
                                y_pred['ccs'][batch['ccs_mask']],
                                batch['ccs'][batch['ccs_mask']],
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
                with_weights=use_weight_vector,
                include_segment_ptr=True,
                weight_tensor_override=weight_tensor,
            )
            loss = loss_fn(y_pred['fragment_probs'], batch[y_tag], **kwargs)
            if not rt_metric:
                safe_metric_update(
                    metric, y_pred['fragment_probs'], batch[y_tag], kwargs
                )
            else:
                if with_rt:
                    safe_metric_update(
                        metric,
                        y_pred['rt'][batch['retention_mask']],
                        batch['retention_time'][batch['retention_mask']],
                        {},
                    )
                if with_ccs:
                    safe_metric_update(
                        metric,
                        y_pred['ccs'][batch['ccs_mask']],
                        batch['ccs'][batch['ccs_mask']],
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

    avg_loss = loss_total / max(loss_batches, 1) if loss_batches > 0 else float('nan')
    metric_label, metric_value = metric_label_and_value(
        metric, preferred_key=metric_name
    )
    return avg_loss, metric_label, metric_value


@dataclass
class EpochResult:
    loss: float
    metric_label: str
    metric_value: float


@dataclass
class TrainingState:
    best_metric: float
    best_epoch: int
    history: dict


def _init_history() -> dict:
    return {
        'epoch': [],
        'train_error': [],
        'sqrt_train_error': [],
        'val_error': [],
        'sqrt_val_error': [],
        'lr': [],
    }


def _record_history(
    history: dict,
    epoch: int,
    lr: float,
    train_result: EpochResult | None,
    val_result: EpochResult | None,
) -> None:
    history['epoch'].append(epoch)
    history['train_error'].append(
        train_result.metric_value if train_result is not None else float('nan')
    )
    history['sqrt_train_error'].append(
        train_result.metric_value if train_result is not None else float('nan')
    )
    history['val_error'].append(
        val_result.metric_value if val_result is not None else float('nan')
    )
    history['sqrt_val_error'].append(
        val_result.metric_value if val_result is not None else float('nan')
    )
    history['lr'].append(lr)


def _run_train_epoch(
    *,
    fabric: Fabric,
    model: torch.nn.Module,
    dataloader,
    loss_fn,
    metric,
    metric_name: str,
    y_label: str,
    with_weights: bool,
    with_rt: bool,
    with_ccs: bool,
    rt_metric: bool,
    optimizer,
    show_progress: bool,
    progress_desc: str,
    non_blocking_transfer: bool,
    precursor_loss_weight: float,
) -> EpochResult:
    loss, label, value = run_epoch(
        fabric=fabric,
        model=model,
        dataloader=dataloader,
        loss_fn=loss_fn,
        metric=metric,
        metric_name=metric_name,
        y_tag=y_label,
        with_weights=with_weights,
        with_rt=with_rt,
        with_ccs=with_ccs,
        rt_metric=rt_metric,
        optimizer=optimizer,
        show_progress=show_progress,
        progress_desc=progress_desc,
        non_blocking_transfer=non_blocking_transfer,
        precursor_loss_weight=precursor_loss_weight,
    )
    return EpochResult(loss=loss, metric_label=label, metric_value=value)


def _run_val_epoch(
    *,
    fabric: Fabric,
    model: torch.nn.Module,
    dataloader,
    loss_fn,
    metric,
    metric_name: str,
    y_label: str,
    with_weights: bool,
    with_rt: bool,
    with_ccs: bool,
    rt_metric: bool,
    use_validation_mask: bool,
    validation_mask_name: str,
    show_progress: bool,
    progress_desc: str,
    non_blocking_transfer: bool,
    precursor_loss_weight: float,
) -> EpochResult:
    loss, label, value = run_epoch(
        fabric=fabric,
        model=model,
        dataloader=dataloader,
        loss_fn=loss_fn,
        metric=metric,
        metric_name=metric_name,
        y_tag=y_label,
        with_weights=with_weights,
        with_rt=with_rt,
        with_ccs=with_ccs,
        rt_metric=rt_metric,
        use_validation_mask=use_validation_mask,
        mask_name=validation_mask_name,
        show_progress=show_progress,
        progress_desc=progress_desc,
        non_blocking_transfer=non_blocking_transfer,
        precursor_loss_weight=precursor_loss_weight,
    )
    return EpochResult(loss=loss, metric_label=label, metric_value=value)


def _monitor_metric(
    *,
    has_validation: bool,
    is_val_cycle: bool,
    train_result: EpochResult,
    val_result: EpochResult | None,
) -> float | None:
    if is_val_cycle and val_result is not None:
        return val_result.metric_value
    if not has_validation:
        return train_result.metric_value
    return None


def _step_scheduler(
    *,
    scheduler,
    optimizer,
    monitor_metric: float | None,
    fabric: Fabric,
    logger: Callable[[str], None] | None,
) -> None:
    if scheduler is None:
        return
    prev_lr = optimizer.param_groups[0]['lr']
    if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
        if monitor_metric is not None and not np.isnan(monitor_metric):
            scheduler.step(monitor_metric)
    else:
        scheduler.step()
    curr_lr = optimizer.param_groups[0]['lr']
    if logger is not None and fabric.is_global_zero and curr_lr < prev_lr:
        logger(f'\t >> Learning rate reduced from {prev_lr:1.0e} to {curr_lr:1.0e}')


def _maybe_update_best(
    *,
    state: TrainingState,
    monitor_metric: float | None,
    epoch: int,
    model,
    output_path: str | None,
    fabric: Fabric,
    logger: Callable[[str], None] | None,
    baseline: bool = False,
) -> None:
    if monitor_metric is None or np.isnan(monitor_metric):
        return
    if monitor_metric >= state.best_metric:
        return
    state.best_metric = monitor_metric
    state.best_epoch = epoch
    if output_path is not None and fabric.is_global_zero:
        unwrap_model(model).save(output_path)
        if logger is not None:
            if baseline:
                logger('\t >> Set baseline checkpoint to epoch 0')
            else:
                logger(f'\t >> Set new checkpoint to epoch {epoch}')


def _log_epoch(
    *,
    epoch: int,
    epochs: int,
    train_result: EpochResult | None,
    val_result: EpochResult | None,
    fabric: Fabric,
    logger: Callable[[str], None] | None,
) -> None:
    if logger is None or not fabric.is_global_zero:
        return

    train_label = (
        train_result.metric_label
        if train_result is not None
        else (val_result.metric_label if val_result is not None else 'metric')
    )
    val_label = (
        val_result.metric_label
        if val_result is not None
        else (train_result.metric_label if train_result is not None else 'metric')
    )

    train_loss_str = (
        f'{train_result.loss:.4f}'
        if train_result is not None and not np.isnan(train_result.loss)
        else 'n/a'
    )
    val_loss_str = (
        f'{val_result.loss:.4f}'
        if val_result is not None and not np.isnan(val_result.loss)
        else 'n/a'
    )
    train_metric_str = (
        f'{train_result.metric_value:.4f}'
        if train_result is not None and not np.isnan(train_result.metric_value)
        else 'n/a'
    )
    val_metric_str = (
        f'{val_result.metric_value:.4f}'
        if val_result is not None and not np.isnan(val_result.metric_value)
        else 'n/a'
    )

    logger(
        f'Epoch {epoch}/{epochs} - '
        f'loss: {train_loss_str} - '
        f'val_loss: {val_loss_str} - '
        f'train_{train_label}: {train_metric_str} - '
        f'val_{val_label}: {val_metric_str}'
    )


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
    precursor_loss_weight: float = 1.0,
):
    has_validation = len(val_data) > 0
    accelerator, devices = resolve_fabric_runtime(device)
    warnings.filterwarnings(
        'ignore',
        message=r'The `srun` command is available on your system but is not used\..*',
        category=PossibleUserWarning,
    )
    if accelerator == 'cuda':
        torch.set_float32_matmul_precision('high')
    if pin_memory is None:
        pin_memory = accelerator == 'cuda'
    use_non_blocking_transfer = bool(pin_memory and accelerator == 'cuda')

    fabric = Fabric(accelerator=accelerator, devices=devices)
    if launch_fabric:
        fabric.launch()

    with_weights = is_weighted_loss(loss_fn)
    if metric_dict:
        metric_name, metric_cls = next(iter(metric_dict.items()))
        train_metric = metric_cls().to(fabric.device)
        val_metric = metric_cls().to(fabric.device)
    else:
        metric_name = 'mse'
        train_metric = MeanSquaredError().to(fabric.device)
        val_metric = MeanSquaredError().to(fabric.device)

    if optimizer is None:
        optimizer = torch.optim.Adam(
            model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )
    if scheduler is None and scheduler_name == 'plateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            patience=scheduler_patience,
            factor=scheduler_factor,
            mode='min',
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

    state = TrainingState(
        best_metric=float('inf'), best_epoch=-1, history=_init_history()
    )

    if has_validation:
        baseline_result = _run_val_epoch(
            fabric=fabric,
            model=model,
            dataloader=val_loader,
            loss_fn=loss_fn,
            metric=val_metric,
            metric_name=metric_name,
            y_label=y_label,
            with_weights=with_weights,
            with_rt=with_rt,
            with_ccs=with_ccs,
            rt_metric=rt_metric,
            use_validation_mask=use_validation_mask,
            validation_mask_name=validation_mask_name,
            show_progress=show_val_progress,
            progress_desc=f'Val 0/{epochs}',
            non_blocking_transfer=use_non_blocking_transfer,
            precursor_loss_weight=precursor_loss_weight,
        )
        if fabric.is_global_zero:
            _record_history(
                state.history,
                epoch=0,
                lr=optimizer.param_groups[0]['lr'],
                train_result=None,
                val_result=baseline_result,
            )
        _maybe_update_best(
            state=state,
            monitor_metric=baseline_result.metric_value,
            epoch=0,
            model=model,
            output_path=output_path,
            fabric=fabric,
            logger=logger,
            baseline=True,
        )
        _log_epoch(
            epoch=0,
            epochs=epochs,
            train_result=None,
            val_result=baseline_result,
            fabric=fabric,
            logger=logger,
        )

    for epoch in range(1, epochs + 1):
        train_result = _run_train_epoch(
            fabric=fabric,
            model=model,
            dataloader=train_loader,
            loss_fn=loss_fn,
            metric=train_metric,
            metric_name=metric_name,
            y_label=y_label,
            with_weights=with_weights,
            with_rt=with_rt,
            with_ccs=with_ccs,
            rt_metric=rt_metric,
            optimizer=optimizer,
            show_progress=show_train_progress,
            progress_desc=f'Train {epoch}/{epochs}',
            non_blocking_transfer=use_non_blocking_transfer,
            precursor_loss_weight=precursor_loss_weight,
        )

        is_val_cycle = has_validation and (epoch % val_every == 0)
        val_result = None
        if is_val_cycle:
            val_result = _run_val_epoch(
                fabric=fabric,
                model=model,
                dataloader=val_loader,
                loss_fn=loss_fn,
                metric=val_metric,
                metric_name=metric_name,
                y_label=y_label,
                with_weights=with_weights,
                with_rt=with_rt,
                with_ccs=with_ccs,
                rt_metric=rt_metric,
                use_validation_mask=use_validation_mask,
                validation_mask_name=validation_mask_name,
                show_progress=show_val_progress,
                progress_desc=f'Val {epoch}/{epochs}',
                non_blocking_transfer=use_non_blocking_transfer,
                precursor_loss_weight=precursor_loss_weight,
            )

        monitor_metric = _monitor_metric(
            has_validation=has_validation,
            is_val_cycle=is_val_cycle,
            train_result=train_result,
            val_result=val_result,
        )
        _step_scheduler(
            scheduler=scheduler,
            optimizer=optimizer,
            monitor_metric=monitor_metric,
            fabric=fabric,
            logger=logger,
        )
        _maybe_update_best(
            state=state,
            monitor_metric=monitor_metric,
            epoch=epoch,
            model=model,
            output_path=output_path,
            fabric=fabric,
            logger=logger,
            baseline=False,
        )

        if (is_val_cycle or not has_validation) and fabric.is_global_zero:
            _record_history(
                state.history,
                epoch=epoch,
                lr=optimizer.param_groups[0]['lr'],
                train_result=train_result,
                val_result=val_result if is_val_cycle else None,
            )
        _log_epoch(
            epoch=epoch,
            epochs=epochs,
            train_result=train_result,
            val_result=val_result,
            fabric=fabric,
            logger=logger,
        )

    if state.best_epoch < 0:
        state.best_epoch = epochs
        state.best_metric = float('nan')
        if output_path is not None and fabric.is_global_zero:
            unwrap_model(model).save(output_path)

    checkpoints = {
        'epoch': state.best_epoch,
        'val_loss': state.best_metric,
        'sqrt_val_loss': state.best_metric,
        'file': output_path,
    }
    return checkpoints, state.history
