r"""Decode a split with a MAS-BIT checkpoint.

    python eval.py --eval_split holdout [--checkpoint weights/mas_bit.safetensors] [--kwargs KEY=VALUE ...]

``holdout`` decodes the hidden competition sets and writes the files the benchmarks accept: one sentence per line for
Brain-to-Text '24 (T12) and ``id,text`` rows for Brain-to-Text '25 (T15).  ``val`` scores the provided validation sets.
Decoding is greedy, in single precision, one trial at a time.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch

from models.mas_bit import AGENTS, MASBIT
from models.text_decoder import load_tokenizer
from utils.checkpoint import load_weights
from utils.config import load_config
from utils.data import PARTICIPANTS, load_datasets
from utils.evaluation import decode, word_error_rate


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", default="weights/mas_bit.safetensors")
    parser.add_argument("--config", default="configs/mas_bit.yaml")
    parser.add_argument("--eval_split", choices=("holdout", "val"), default="holdout")
    parser.add_argument("--participants", nargs="+", choices=PARTICIPANTS, default=list(PARTICIPANTS))
    parser.add_argument("--output_dir", default=None, help="default: outputs/<eval_split>")
    parser.add_argument("--kwargs", nargs="*", default=[], help="config overrides, e.g. paths.data_dir=/data")
    args = parser.parse_args()
    config = load_config(args.config, args.kwargs)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = load_tokenizer(config.paths.text_decoder_dir)
    # Built as in training (bfloat16 text decoder, so its rotary frequencies are bfloat16-rounded), then cast to
    # float32: this reproduces the released holdout predictions exactly.
    model = MASBIT(config, tokenizer).float()
    load_weights(args.checkpoint, model)
    model.to(device).eval()

    output = Path(args.output_dir or Path("outputs") / args.eval_split)
    output.mkdir(parents=True, exist_ok=True)
    dataset = load_datasets(config.paths.data_dir, tokenizer, args.participants)[args.eval_split]
    records = decode(model, dataset, tokenizer, device)
    with (output / "predictions.jsonl").open("w") as stream:
        stream.writelines(json.dumps(record) + "\n" for record in records)

    for participant in args.participants:
        rows = [record for record in records if record["participant"] == participant]
        if args.eval_split == "val":
            scores = {agent: word_error_rate(rows, agent) for agent in AGENTS}
            (output / f"{participant}_wer.json").write_text(json.dumps(scores, indent=2) + "\n")
            print(
                f"{participant.upper()} validation WER: "
                + ", ".join(f"{agent} {100 * scores[agent]['wer']:.3f}%" for agent in AGENTS)
            )
        elif participant == "t12":
            (output / "t12_holdout.txt").write_text("".join(row["arbiter"] + "\n" for row in rows))
        else:
            with (output / "t15_holdout.csv").open("w", newline="") as stream:
                writer = csv.writer(stream)
                writer.writerow(["id", "text"])
                writer.writerows((i, row["arbiter"]) for i, row in enumerate(rows))
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
