"""Brain-to-Text '24 (participant T12, Willett et al., 2023): threshold crossings and spike-band power of area 6v."""

from __future__ import annotations

import os
import string
from glob import glob

import numpy as np
import scipy.io
from tqdm import tqdm

SPLIT_DIRS = {"train": "train", "val": "test", "holdout": "competitionHoldOut"}
FEATURES = ("tx1", "spikePow")
#: channels of area 6v within each feature
AREA = slice(0, 128)
KEEP_PUNCTUATION = str.maketrans("", "", string.punctuation.replace("'", ""))


def session_date(file: str) -> tuple[str, ...]:
    return tuple(os.path.basename(file).split(".")[1:4])


def zscore_by_block(trials: list[np.ndarray], blocks: np.ndarray) -> list[np.ndarray]:
    for block in set(blocks.flatten().tolist()):
        index = np.where(blocks == block)[0]
        stacked = np.concatenate([trials[i] for i in index], axis=0)
        mean, std = np.mean(stacked, axis=0), np.std(stacked, axis=0)
        for i in index:
            trials[i] = (trials[i] - mean) / std
    return trials


def load_session(file: str) -> list[dict]:
    data = scipy.io.loadmat(file)
    sentences, blocks = data["sentenceText"], data["blockIdx"]
    trials = [
        np.concatenate([data[feature][0, i][:, AREA] for feature in FEATURES], axis=1) for i in range(len(sentences))
    ]
    trials = zscore_by_block(trials, blocks)
    return [
        {
            "spikes": spikes.astype(np.float32),
            "sentence": str(sentence).translate(KEEP_PUNCTUATION).lower().strip(),
            "day": session_date(file),
        }
        for spikes, sentence in zip(trials, sentences)
    ]


def zscore_by_day(dataset: dict) -> dict:
    """Normalise every session with the mean and standard deviation of its training trials."""
    days = sorted({row["day"] for rows in dataset.values() for row in rows})
    stats = {}
    for day in days:
        stacked = np.concatenate([row["spikes"] for row in dataset["train"] if row["day"] == day], axis=0)
        stats[day] = (np.mean(stacked, axis=0), np.std(stacked, axis=0))
    for rows in dataset.values():
        for row in rows:
            mean, std = stats[row["day"]]
            row["spikes"] = (row["spikes"] - mean) / std
    return dataset


def load_data(data_dir: str) -> dict[str, list[dict]]:
    """``{"train", "val", "holdout"}``: lists of trials with ``spikes`` (frames, 256), ``sentence`` and ``day``.
    ``val`` is the labelled ``test`` directory of the release; ``holdout`` is the competition's hidden set (no
    sentences)."""
    dataset = {}
    for split, directory in SPLIT_DIRS.items():
        files = sorted(glob(os.path.join(data_dir, directory, "*")), key=session_date)
        dataset[split] = [row for file in tqdm(files, desc=f"T12 {split}") for row in load_session(file)]
    for row in dataset["holdout"]:
        row["sentence"] = None  # the release fills the hidden sentences with the placeholder "held out"
    return zscore_by_day(dataset)
