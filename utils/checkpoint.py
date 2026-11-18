"""Checkpoints.  A MAS-BIT checkpoint is one ``.safetensors`` file holding the weights MAS-BIT adds to the frozen text
decoder: the encoders, the interfaces, the phoneme readouts and the three adapters.  ``configs/mas_bit.yaml`` sets the
architecture.  To resume training, ``train.py`` also keeps the optimizer states and its progress in a ``.pt`` file."""

from __future__ import annotations

import torch
from safetensors.torch import load_file, save_file


def save_weights(path, model):
    save_file(model.adaptation_state_dict(), str(path))


def load_weights(path, model):
    model.load_adaptation_state_dict(load_file(str(path)))


def save_training_state(path, state: dict):
    torch.save(state, path)


def load_training_state(path) -> dict:
    return torch.load(path, map_location="cpu", weights_only=False)
