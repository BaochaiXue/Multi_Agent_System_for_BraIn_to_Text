"""Batch sampler of joint training: every trial once per epoch, batches of one participant and of similar length."""

from __future__ import annotations

import numpy as np


class TrainBatchSampler:
    """Yields the batches of one GPU (``rank`` of ``world_size``).

    Per epoch, the trials of each participant are shuffled, sorted by length inside windows of
    ``bucket_window * batch_size`` trials and cut into batches; batches of all participants are shuffled together.
    Batches of similar cost (longest trial x trials) then share a distributed step.  If the last step is incomplete it
    is filled with repeated batches that are flagged inactive (``self.active``), so that all GPUs run the same number
    of steps while every trial still counts exactly once.
    """

    def __init__(
        self,
        participants: list[str],
        lengths: list[int],
        batch_size: int,
        bucket_window: int,
        rank: int = 0,
        world_size: int = 1,
        seed: int = 0,
    ):
        self.lengths, self.batch_size, self.window = lengths, batch_size, bucket_window
        self.rank, self.world_size, self.seed, self.epoch = rank, world_size, seed, 0
        self.groups = {}
        for index, participant in enumerate(participants):
            self.groups.setdefault(participant, []).append(index)
        self.rng = np.random.default_rng(seed)
        self.active: list[bool] = []

    def __len__(self) -> int:
        batches = sum(-(-len(indices) // self.batch_size) for indices in self.groups.values())
        return -(-batches // self.world_size)

    def epoch_batches(self) -> list[list[int]]:
        batches, span = [], self.window * self.batch_size
        for indices in self.groups.values():
            indices = np.array(indices)
            self.rng.shuffle(indices)
            chunks = [indices[start : start + span] for start in range(0, len(indices), span)]
            indices = np.concatenate(
                [chunk[np.argsort([-self.lengths[i] for i in chunk], kind="stable")] for chunk in chunks]
            )
            batches.extend(
                indices[start : start + self.batch_size].tolist() for start in range(0, len(indices), self.batch_size)
            )
        self.rng.shuffle(batches)
        return batches

    def epoch_plan(self) -> list[tuple[list[int], bool]]:
        """(batch, active) of every GPU slot of the epoch; slot ``s`` belongs to GPU ``s % world_size``."""
        batches = self.epoch_batches()
        self.epoch += 1
        rng = np.random.default_rng(self.seed * 104729 + self.epoch)
        cost = [max(self.lengths[i] for i in batch) * len(batch) for batch in batches]
        order = (
            sorted(range(len(batches)), key=lambda b: (-cost[b], b))
            if self.world_size > 1
            else list(range(len(batches)))
        )
        steps = [order[start : start + self.world_size] for start in range(0, len(order), self.world_size)]
        if self.world_size > 1:
            rng.shuffle(steps)
        plan = []
        for step in steps:
            slots = [(batches[b], True) for b in step]
            if len(slots) < self.world_size:
                filler = batches[min(step, key=lambda b: (cost[b], b))]
                slots += [(filler, False)] * (self.world_size - len(slots))
            if self.world_size > 1:
                slots = [slots[i] for i in rng.permutation(len(slots))]
            plan.extend(slots)
        return plan

    def __iter__(self):
        plan = self.epoch_plan()[self.rank :: self.world_size]
        self.active = [active for _, active in plan]
        yield from (batch for batch, _ in plan)

    def skip_epochs(self, epochs: int):
        """Advance the sampler to the state after ``epochs`` epochs (resuming a run)."""
        for _ in range(epochs):
            self.epoch_batches()
        self.epoch += epochs
