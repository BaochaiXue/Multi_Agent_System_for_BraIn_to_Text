"""Staged-to-joint training schedule: which parameter groups train in an epoch, and their learning rates."""

from __future__ import annotations

import math
from dataclasses import dataclass


def warmup_cosine(step: int, steps: int, warmup: float, floor: float) -> float:
    """Scale of the peak learning rate at update ``step`` of a stage with ``steps`` updates: linear warm-up from
    ``floor`` over the first ``warmup`` fraction, then cosine decay back to ``floor``."""
    step = max(0, min(step, steps))
    warm = max(1, int(round(warmup * steps)))
    if step < warm:
        return floor + (1.0 - floor) * step / warm
    progress = (step - warm) / max(1, steps - warm)
    return floor + (1.0 - floor) * 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))


@dataclass
class Position:
    stage: dict
    epoch_in_stage: int


class StageSchedule:
    def __init__(self, training, optimizer, steps_per_epoch: int):
        self.stages, self.optimizer, self.steps_per_epoch = training.stages, optimizer, steps_per_epoch
        self.total_epochs = sum(stage.epochs for stage in self.stages)

    def position(self, epoch: int) -> Position:
        """Stage of the one-based ``epoch``."""
        first = 1
        for stage in self.stages:
            if epoch < first + stage.epochs:
                return Position(stage, epoch - first + 1)
            first += stage.epochs
        raise ValueError(f"Epoch {epoch} is beyond the {self.total_epochs}-epoch schedule")

    def trains_encoder(self, epoch: int) -> bool:
        return "encoder" in self.position(epoch).stage.train

    def learning_rate(self, group: str, epoch: int, step_in_epoch: int, muon: bool = False) -> float:
        """Learning rate of ``group`` for update ``step_in_epoch`` (zero-based) of ``epoch``; zero if the stage does
        not train it.  ``muon`` selects the schedule of the encoder weight matrices."""
        position = self.position(epoch)
        stage = position.stage
        if group not in stage.train:
            return 0.0
        step = (position.epoch_in_stage - 1) * self.steps_per_epoch + step_in_epoch
        steps = stage.epochs * self.steps_per_epoch
        if muon:
            peak, warmup = self.optimizer.muon.lr, stage.get("muon_warmup", self.optimizer.muon.warmup)
        else:
            peak, warmup = stage.get("lr", {}).get(group, self.optimizer.lr[group]), self.optimizer.warmup
        return peak * stage.get("lr_scale", 1.0) * warmup_cosine(step, steps, warmup, self.optimizer.floor)

    def intermediate_ctc_scale(self, epoch: int, forwards_in_epoch: int, forwards_per_epoch: int) -> float:
        """Ramp of the intermediate CTC loss: zero until the encoders train, linear over their first epoch, then one."""
        completed = sum(self.trains_encoder(e) for e in range(1, epoch))
        partial = forwards_in_epoch / forwards_per_epoch if self.trains_encoder(epoch) else 0.0
        return min(1.0, completed + partial)
