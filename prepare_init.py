r"""Build the initial weights of MAS-BIT from the released BIT checkpoints of participants T12 and T15.

    python prepare_init.py --t12_checkpoint YOUR_BIT_T12_CHECKPOINT/model.safetensors \
                           --t15_checkpoint YOUR_BIT_T15_CHECKPOINT/model.safetensors

From each checkpoint this keeps the neural encoder, the input projection of the checkpoint's own participant and its
phoneme readout, and writes them to ``weights/init.safetensors`` (``paths.init_checkpoint``).  The input projections
for the other participant, the interfaces and the adapters are new modules and are initialised by ``train.py``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from safetensors import safe_open
from safetensors.torch import save_file

#: index of each participant's dataset among the input projections and phoneme readouts of a BIT checkpoint
BIT_DATASET_INDEX = {"t15": 0, "t12": 1}
ENCODER = "neural_encoder.encoder."
INPUT_PROJECTION = ENCODER + "embedder.read_in.embed_dict.{index}."
PHONEME_READOUT = "neural_encoder.decoder.read_out.readout_dict.{index}."


def encoder_state(checkpoint: str, participant: str) -> dict:
    index = BIT_DATASET_INDEX[participant]
    projection, readout = INPUT_PROJECTION.format(index=index), PHONEME_READOUT.format(index=index)
    state = {}
    with safe_open(checkpoint, framework="pt", device="cpu") as tensors:
        for key in tensors.keys():
            if key.startswith(projection):
                name = f"encoders.{participant}.input_projections.{participant}.{key.removeprefix(projection)}"
            elif key.startswith(ENCODER + "layers.") or key.startswith(ENCODER + "out_norm."):
                name = f"encoders.{participant}.{key.removeprefix(ENCODER)}"
            elif key == ENCODER + "embedder.mask_token":
                name = f"encoders.{participant}.mask_token"
            elif key.startswith(readout):
                name = f"phoneme_heads.{participant}.{key.removeprefix(readout)}"
            else:
                continue
            state[name] = tensors.get_tensor(key).float()
    return state


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--t12_checkpoint", required=True, help="model.safetensors of the released BIT T12 checkpoint")
    parser.add_argument("--t15_checkpoint", required=True, help="model.safetensors of the released BIT T15 checkpoint")
    parser.add_argument("--output", default="weights/init.safetensors")
    args = parser.parse_args()
    state = {**encoder_state(args.t12_checkpoint, "t12"), **encoder_state(args.t15_checkpoint, "t15")}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    save_file(state, str(output))
    print(f"Wrote {len(state)} tensors to {output}")


if __name__ == "__main__":
    main()
