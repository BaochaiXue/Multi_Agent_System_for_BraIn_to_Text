"""Trials of both participants: loading, text and phoneme labels, and batching."""

from __future__ import annotations

import importlib
import json
import os
import re
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

DATASETS = {"t12": "willett_2023_text", "t15": "brandman_2024_text"}
PARTICIPANTS = tuple(DATASETS)
VOCAB_FILE = Path(__file__).resolve().parents[1] / "vocab.json"
IGNORE = -100


def normalize_sentence(text: str) -> str:
    """Lower-case words without punctuation, as the benchmarks score them."""
    text = re.sub(r"[^a-zA-Z\- ']+", "", text).replace("- ", " ").lower()
    text = text.replace("--", "").replace(" '", "'")
    return " ".join(text.split())


def phoneme_ids(sentences: list[str]) -> list[np.ndarray]:
    """CTC targets: the phonemes of every word followed by a silence token (``vocab.json``)."""
    from g2p_en import G2p

    g2p, vocab = G2p(), json.loads(VOCAB_FILE.read_text())
    ids = []
    for sentence in sentences:
        phonemes = [
            re.sub(r"[0-9]", "", p) if p != " " else "SIL" for p in g2p(sentence) if re.match(r"[A-Z]+", p) or p == " "
        ]
        ids.append(np.asarray([vocab.index(p) for p in phonemes + ["SIL"]]))
    return ids


def load_participant(participant: str, data_dir: str, tokenizer) -> dict[str, list[dict]]:
    """``{"train", "val", "holdout"}`` trials of one participant.  Labelled trials carry ``targets`` (token ids of the
    sentence and the end-of-sequence token) and ``phonemes``; holdout trials have no sentence."""
    name = DATASETS[participant]
    dataset = importlib.import_module(f"data.{name}.prepare_data").load_data(os.path.join(data_dir, name))
    for rows in dataset.values():
        labelled = [row for row in rows if row["sentence"]]
        for row, phonemes in zip(labelled, phoneme_ids([row["sentence"] for row in labelled])):
            row["targets"] = tokenizer(row["sentence"] + tokenizer.eos_token, return_tensors="np")["input_ids"][0]
            row["phonemes"] = phonemes
        for row in rows:
            row["participant"] = participant
    return dataset


def load_datasets(data_dir: str, tokenizer, participants=PARTICIPANTS) -> dict[str, TrialDataset]:
    """One dataset per split, holding the trials of the requested participants (T12 first)."""
    parts = {participant: load_participant(participant, data_dir, tokenizer) for participant in participants}
    return {
        split: TrialDataset([row for participant in participants for row in parts[participant][split]])
        for split in ("train", "val", "holdout")
    }


class TrialDataset(Dataset):
    def __init__(self, rows: list[dict]):
        self.rows = rows
        self.participants = [row["participant"] for row in rows]
        self.lengths = [int(row["spikes"].shape[0]) for row in rows]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        return {**self.rows[index], "index": index}


def collate(rows: list[dict]) -> dict:
    """Right-pad the trials of one participant to a batch."""
    participants = {row["participant"] for row in rows}
    if len(participants) != 1:
        raise ValueError("A batch holds trials of one participant")
    frames = max(row["spikes"].shape[0] for row in rows)
    spikes = torch.zeros(len(rows), frames, rows[0]["spikes"].shape[1])
    mask = torch.zeros(len(rows), frames, dtype=torch.long)
    timestamp = torch.zeros(len(rows), frames, dtype=torch.long)
    for i, row in enumerate(rows):
        length = row["spikes"].shape[0]
        spikes[i, :length], mask[i, :length] = torch.from_numpy(row["spikes"]), 1
        timestamp[i, :length] = torch.arange(length)
    batch = {
        "spikes": spikes,
        "spikes_mask": mask,
        "spikes_timestamp": timestamp,
        "participant": participants.pop(),
        "index": [row["index"] for row in rows],
        "sentence": [row["sentence"] for row in rows],
    }
    if all("targets" in row for row in rows):
        batch.update(targets=pad_labels(rows, "targets", IGNORE), phonemes=pad_labels(rows, "phonemes", -1))
    return batch


def pad_labels(rows: list[dict], key: str, value: int) -> torch.Tensor:
    labels = [torch.from_numpy(row[key]).long() for row in rows]
    return torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=value)


MODEL_INPUTS = ("spikes", "spikes_mask", "spikes_timestamp", "targets", "phonemes")


def to_device(batch: dict, device, keys=MODEL_INPUTS) -> dict:
    """The tensors the model reads, on ``device``, plus the participant."""
    return {"participant": batch["participant"], **{key: batch[key].to(device) for key in keys if key in batch}}
