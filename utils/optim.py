"""Optimizers of MAS-BIT: Muon for the weight matrices of the encoder blocks, AdamW for everything else."""

from __future__ import annotations

import torch

from models.mas_bit import parameter_group


def is_encoder_matrix(name: str, parameter) -> bool:
    return (
        parameter_group(name) == "encoder" and ".layers." in name and name.endswith(".weight") and parameter.ndim == 2
    )


def build_optimizers(model, config):
    """One parameter group per (model parameter group, weight decay); ``group["name"]`` is the model parameter group."""
    adamw, muon = {}, []
    for name, parameter in model.adaptation_parameters():
        if is_encoder_matrix(name, parameter):
            muon.append(parameter)
            continue
        decay = not (name.endswith(".bias") or parameter.ndim <= 1)
        adamw.setdefault((parameter_group(name), decay), []).append(parameter)
    options = config.adamw
    adamw_groups = [
        {
            "name": group,
            "params": parameters,
            "lr": config.lr[group],
            "weight_decay": options.weight_decay if decay else 0.0,
        }
        for (group, decay), parameters in sorted(adamw.items())
    ]
    muon_groups = [{"name": "encoder", "params": muon, "lr": config.muon.lr, "weight_decay": config.muon.weight_decay}]
    return {
        "muon": torch.optim.Muon(
            muon_groups,
            lr=config.muon.lr,
            momentum=config.muon.momentum,
            nesterov=True,
            ns_steps=config.muon.ns_steps,
            adjust_lr_fn="match_rms_adamw",
        ),
        "adamw": torch.optim.AdamW(adamw_groups, lr=config.lr.encoder, betas=tuple(options.betas), eps=options.eps),
    }


def set_learning_rates(optimizers, schedule, epoch: int, step_in_epoch: int):
    for kind, optimizer in optimizers.items():
        for group in optimizer.param_groups:
            group["lr"] = schedule.learning_rate(group["name"], epoch, step_in_epoch, muon=kind == "muon")


def clip_gradients(optimizers, max_norm: float) -> torch.Tensor:
    """Clip the global gradient norm of the parameters that received a gradient in this update."""
    parameters = [
        p
        for optimizer in optimizers.values()
        for group in optimizer.param_groups
        for p in group["params"]
        if p.grad is not None
    ]
    return torch.nn.utils.clip_grad_norm_(parameters, max_norm, error_if_nonfinite=True)
