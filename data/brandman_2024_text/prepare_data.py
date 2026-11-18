"""Brain-to-Text '25 (participant T15, Card et al., 2024): threshold crossings and spike-band power of four arrays."""

from __future__ import annotations

import os
import string

import h5py
import numpy as np
from tqdm import tqdm

SPLIT_FILES = {"train": "data_train.hdf5", "val": "data_val.hdf5", "holdout": "data_test.hdf5"}
KEEP_PUNCTUATION = str.maketrans("", "", string.punctuation.replace("'", ""))


def session_date(session: str) -> tuple[str, ...]:
    return tuple(session.split(".")[1:4])


def load_session(file: str, day: tuple[str, ...]) -> list[dict]:
    rows = []
    with h5py.File(file, "r") as trials:
        for key in trials.keys():
            trial = trials[key]
            sentence = trial.attrs["sentence_label"][:] if "sentence_label" in trial.attrs else None
            rows.append(
                {
                    "spikes": trial["input_features"][:].astype(np.float32),
                    "sentence": None if sentence is None else sentence.translate(KEEP_PUNCTUATION).lower().strip(),
                    "day": day,
                }
            )
    return rows


def load_data(data_dir: str) -> dict[str, list[dict]]:
    """``{"train", "val", "holdout"}``: lists of trials with ``spikes`` (frames, 512), ``sentence`` and ``day``.
    The released features are already normalised.  ``holdout`` is the competition's hidden set (no sentences)."""
    root = os.path.join(data_dir, "hdf5_data_final")
    sessions = sorted((entry.name for entry in os.scandir(root) if entry.is_dir()), key=session_date)
    dataset = {}
    for split, name in SPLIT_FILES.items():
        files = [(os.path.join(root, session, name), session_date(session)) for session in sessions]
        dataset[split] = [
            row
            for file, day in tqdm(files, desc=f"T15 {split}")
            if os.path.isfile(file)
            for row in load_session(file, day)
        ]
    return dataset
