import math

import torch

from fiora.GNN.Trainer import Trainer


class _DummyTrainer(Trainer):
    def _training_loop(self, model, dataloader, optimizer, loss_fn, **kwargs):
        return None

    def _validation_loop(self, model, dataloader, loss_fn, **kwargs):
        return None

    def train(self, model, optimizer, loss_fn, **kwargs):
        return None


def test_update_history_supports_mae_only_stats():
    trainer = _DummyTrainer(data=[], only_training=True)
    trainer._init_history()
    trainer._update_history(
        epoch=1,
        train_stats={"mae": torch.tensor(0.5)},
        val_stats={"mae": torch.tensor(0.75)},
        lr=1e-3,
    )

    assert trainer.history["train_error"] == [0.5]
    assert trainer.history["val_error"] == [0.75]
    assert math.isnan(trainer.history["sqrt_train_error"][0])
    assert math.isnan(trainer.history["sqrt_val_error"][0])
    assert trainer.history["lr"] == [1e-3]
