"""Greedy decoding of a split and word error rate."""

from __future__ import annotations

import editdistance
import torch
from tqdm import tqdm

from models.text_decoder import AGENTS
from utils.data import collate, normalize_sentence, to_device


def word_errors(prediction: str, reference: str) -> tuple[int, int]:
    """(word edit distance, number of reference words)."""
    reference = reference.split()
    return int(editdistance.eval(prediction.split(), reference)), len(reference)


@torch.inference_mode()
def decode(
    model, dataset, tokenizer, device, indices=None, autocast: bool = False, progress: bool = True
) -> list[dict]:
    """Decode the trials ``indices`` of ``dataset`` one at a time.  Every record holds the normalised sentence of each
    agent (``arbiter`` is the system's output) and, for labelled trials, the reference and each agent's word errors."""
    model.eval()
    records = []
    for index in tqdm(range(len(dataset)) if indices is None else indices, desc="Decoding", disable=not progress):
        batch = collate([dataset[index]])
        inputs = to_device(batch, device, keys=("spikes", "spikes_mask", "spikes_timestamp"))
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=autocast):
            generated = model.generate(**inputs)
        record = {
            "index": index,
            "participant": batch["participant"],
            **{
                agent: normalize_sentence(tokenizer.decode(generated[agent][0], skip_special_tokens=True))
                for agent in AGENTS
            },
        }
        if batch["sentence"][0]:
            record["reference"] = normalize_sentence(batch["sentence"][0])
            for agent in AGENTS:
                record[f"{agent}_errors"], record["words"] = word_errors(record[agent], record["reference"])
        records.append(record)
    return records


def word_error_rate(records: list[dict], agent: str = "arbiter") -> dict:
    """Corpus word error rate of ``agent``: total word errors over total reference words."""
    errors, words = sum(r[f"{agent}_errors"] for r in records), sum(r["words"] for r in records)
    return {"errors": errors, "words": words, "sentences": len(records), "wer": errors / max(words, 1)}
