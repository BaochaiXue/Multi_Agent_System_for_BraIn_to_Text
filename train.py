"""Train MAS-BIT jointly on participants T12 and T15.

    torchrun --nproc_per_node 8 train.py --config configs/mas_bit.yaml

Writes to ``paths.output_dir``:

    config.yaml           configuration of the run
    history.jsonl         training losses and validation WER of every epoch
    best.safetensors      checkpoint with the fewest arbiter word errors on the pooled validation sets
    last.safetensors      checkpoint of the latest epoch
    training_state.pt     optimizer states and progress of the latest epoch, for ``--resume``
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from safetensors.torch import load_file
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader

from models.mas_bit import AGENTS, IGNORE, MASBIT
from models.text_decoder import load_tokenizer
from utils.checkpoint import load_training_state, load_weights, save_training_state, save_weights
from utils.config import load_config, save_config
from utils.data import collate, load_datasets, to_device
from utils.evaluation import decode
from utils.optim import build_optimizers, clip_gradients, set_learning_rates
from utils.sampler import TrainBatchSampler
from utils.schedule import StageSchedule


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def all_reduce_sum(values: list[float], device) -> list[float]:
    tensor = torch.tensor(values, dtype=torch.float64, device=device)
    if dist.is_initialized():
        dist.all_reduce(tensor)
    return tensor.tolist()


def load_initialization(model, path: str):
    """Released BIT encoders with their own input projections and phoneme readouts (``prepare_init.py``); the input
    projections for the other participant, the interfaces and the adapters start from their random initialisation."""
    state = load_file(path)
    fresh = ("encoders.t12.input_projections.t15.", "encoders.t15.input_projections.t12.")
    expected = {
        name
        for name in model.adaptation_state_dict()
        if name.startswith(("encoders.", "phoneme_heads.")) and not name.startswith(fresh)
    }
    if set(state) != expected:
        raise ValueError(f"{path} does not hold the released BIT encoders and phoneme readouts")
    model.load_state_dict(state, strict=False)


def update_windows(loader, sampler, grad_accum: int):
    """The batches of each optimizer update, with the flags of the sampler's inactive filler batches."""
    window = []
    for forward, batch in enumerate(loader):
        window.append((batch, sampler.active[forward]))
        if len(window) == grad_accum:
            yield window
            window = []
    if window:
        yield window


def validate(model, dataset, tokenizer, device, rank: int, world_size: int) -> dict:
    """Pooled validation word errors of every agent; each GPU decodes its share of the trials."""
    records = decode(
        model,
        dataset,
        tokenizer,
        device,
        indices=range(rank, len(dataset), world_size),
        autocast=device.type == "cuda",
        progress=rank == 0,
    )
    totals = all_reduce_sum(
        [sum(r[f"{agent}_errors"] for r in records) for agent in AGENTS] + [sum(r["words"] for r in records)], device
    )
    return {"words": int(totals[-1]), **{agent: int(errors) for agent, errors in zip(AGENTS, totals)}}


def train_epoch(model, wrapped, loader, sampler, optimizers, schedule, config, epoch: int, device, log) -> dict:
    training = config.training
    stage = schedule.position(epoch).stage
    model.train()
    for optimizer in optimizers.values():
        optimizer.zero_grad(set_to_none=True)
    sums = torch.zeros(len(AGENTS) + 4, dtype=torch.float64, device=device)
    for update, window in enumerate(update_windows(loader, sampler, training.grad_accum)):
        # the objective of an update is normalised by the text tokens and phonemes of all its batches
        text_tokens = sum(int((batch["targets"] != IGNORE).sum()) for batch, active in window if active) or 1
        phonemes = sum(int((batch["phonemes"] >= 0).sum()) for batch, active in window if active) or 1
        for offset, (batch, active) in enumerate(window):
            forward = update * training.grad_accum + offset + 1
            ramp = schedule.intermediate_ctc_scale(epoch, forward, len(sampler))
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                losses = wrapped(**to_device(batch, device))
            cross_entropy = sum(losses.cross_entropy.values())
            loss = (
                cross_entropy / text_tokens
                + training.ctc_weight * losses.ctc / phonemes
                + training.intermediate_ctc_weight * ramp * losses.intermediate_ctc / phonemes
            )
            (loss * float(active)).backward()
            sums += float(active) * torch.stack(
                [
                    *(losses.cross_entropy[agent].detach().double() for agent in AGENTS),
                    losses.ctc.detach().double(),
                    losses.intermediate_ctc.detach().double(),
                    losses.text_tokens.double(),
                    losses.phonemes.double(),
                ]
            )
        clip_gradients(optimizers, training.grad_clip)
        set_learning_rates(optimizers, schedule, epoch, update)
        for optimizer in optimizers.values():
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        if update % 50 == 0:
            log(f"epoch {epoch} [{stage.name}] update {update + 1}/{schedule.steps_per_epoch}")
    *cross_entropy, ctc, intermediate_ctc, text_tokens, phonemes = all_reduce_sum(sums.tolist(), device)
    return {
        **{f"cross_entropy_{agent}": value / text_tokens for agent, value in zip(AGENTS, cross_entropy)},
        "ctc": ctc / phonemes,
        "intermediate_ctc": intermediate_ctc / phonemes,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="configs/mas_bit.yaml")
    parser.add_argument("--resume", action="store_true", help="continue from the latest epoch in paths.output_dir")
    parser.add_argument("--kwargs", nargs="*", default=[], help="config overrides, e.g. paths.output_dir=runs/a seed=1")
    args = parser.parse_args()
    config = load_config(args.config, args.kwargs)

    distributed = "LOCAL_RANK" in os.environ
    if distributed:
        dist.init_process_group("nccl" if torch.cuda.is_available() else "gloo")
    rank, world_size = (dist.get_rank(), dist.get_world_size()) if distributed else (0, 1)
    device = (
        torch.device("cuda", int(os.environ.get("LOCAL_RANK", 0))) if torch.cuda.is_available() else torch.device("cpu")
    )
    seed_everything(config.seed)
    output = Path(config.paths.output_dir)
    log = (lambda message: print(message, flush=True)) if rank == 0 else (lambda message: None)

    tokenizer = load_tokenizer(config.paths.text_decoder_dir)
    datasets = load_datasets(config.paths.data_dir, tokenizer)
    train_set, val_set = datasets["train"], datasets["val"]
    sampler = TrainBatchSampler(
        train_set.participants,
        train_set.lengths,
        config.training.batch_size,
        config.training.bucket_window,
        rank,
        world_size,
        config.seed,
    )
    loader = DataLoader(train_set, batch_sampler=sampler, collate_fn=collate)
    schedule = StageSchedule(
        config.training, config.optimizer, steps_per_epoch=-(-len(sampler) // config.training.grad_accum)
    )

    model = MASBIT(config, tokenizer).to(device)
    load_initialization(model, config.paths.init_checkpoint)
    optimizers = build_optimizers(model, config.optimizer)
    state = {"epoch": 0, "best_epoch": None, "best_errors": None}
    if args.resume:
        load_weights(output / "last.safetensors", model)
        saved = load_training_state(output / "training_state.pt")
        for name, optimizer in optimizers.items():
            optimizer.load_state_dict(saved["optimizers"][name])
        state = saved["state"]
        sampler.skip_epochs(state["epoch"])
    elif rank == 0:
        output.mkdir(parents=True, exist_ok=True)
        save_config(config, output / "config.yaml")
    log(
        f"{len(train_set)} training and {len(val_set)} validation trials; {world_size} GPU(s); "
        f"{schedule.steps_per_epoch} updates per epoch for {schedule.total_epochs} epochs"
    )

    selection, stage_name, wrapped = config.training.selection, None, model
    for epoch in range(state["epoch"] + 1, schedule.total_epochs + 1):
        stage = schedule.position(epoch).stage
        if stage.name != stage_name:
            # the set of parameters that receive gradients changes with the stage
            stage_name = stage.name
            model.set_trainable(set(stage.train))
            device_ids = [device.index] if device.type == "cuda" else None
            wrapped = (
                DistributedDataParallel(model, device_ids=device_ids, find_unused_parameters=True)
                if distributed
                else model
            )
        started = time.time()
        losses = train_epoch(model, wrapped, loader, sampler, optimizers, schedule, config, epoch, device, log)
        errors = validate(model, val_set, tokenizer, device, rank, world_size)
        state["epoch"] = epoch
        improved = epoch >= selection.start_epoch and (
            state["best_errors"] is None or errors["arbiter"] < state["best_errors"]
        )
        if improved:
            state.update(best_epoch=epoch, best_errors=errors["arbiter"])
        if rank == 0:
            record = {
                "epoch": epoch,
                "stage": stage.name,
                **losses,
                **{f"val_wer_{agent}": errors[agent] / errors["words"] for agent in AGENTS},
                "val_errors_arbiter": errors["arbiter"],
                "best_epoch": state["best_epoch"],
                "minutes": (time.time() - started) / 60,
            }
            log(json.dumps(record))
            with (output / "history.jsonl").open("a") as history:
                history.write(json.dumps(record) + "\n")
            save_weights(output / "last.safetensors", model)
            optimizer_states = {name: optimizer.state_dict() for name, optimizer in optimizers.items()}
            save_training_state(output / "training_state.pt", {"state": state, "optimizers": optimizer_states})
            if improved:
                save_weights(output / "best.safetensors", model)
        if state["best_epoch"] is not None and epoch - state["best_epoch"] >= selection.patience:
            log(f"No improvement for {selection.patience} epochs; best epoch {state['best_epoch']}")
            break
    best = state["best_epoch"]
    log(f"Best checkpoint: epoch {best} ({state['best_errors']} validation word errors): {output / 'best.safetensors'}")
    if distributed:
        dist.barrier()  # every GPU waits until the last checkpoint is written
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
