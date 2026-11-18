## MAS-BIT: A Multi-Agent System for End-to-End BraIn-to-Text Decoding

**MAS-BIT** decodes sentences from intracortical neural activity with three language agents that share one frozen audio-LLM
text decoder. A *neural expert* reads continuous encoder representations, a *phonetic expert* reads phoneme posteriors,
and an *arbiter* combines their latent messages with direct neural evidence to produce the sentence. One model is trained
jointly on participants T12 (Brain-to-Text '24) and T15 (Brain-to-Text '25) and decodes greedily, without a lexicon,
an n-gram model or an external language-model reranker.

- [Setup](#setup)
- [Training](#training)
- [Inference with released weights](#inference-with-released-weights)
- [Code structure](#code-structure)

## Setup

```bash
conda env create -f env.yaml
conda activate mas-bit
python -c "import nltk; nltk.download('averaged_perceptron_tagger_eng'); nltk.download('cmudict')"
```

The last line fetches the pronunciation dictionary and part-of-speech tagger used to derive the phoneme labels.

#### Data

Download both benchmarks into one folder (`YOUR_DATA_DIR`):

- **Brain-to-Text '24 (T12)**: `competitionData.tar.gz` from [DRYAD](https://datadryad.org/dataset/doi:10.5061/dryad.x69p8czpq),
  extracted and renamed to `willett_2023_text`.
- **Brain-to-Text '25 (T15)**: the data from [DRYAD](https://datadryad.org/dataset/doi:10.5061/dryad.dncjsxm85), renamed to
  `brandman_2024_text`.

Then set this folder in `configs/mas_bit.yaml`:

```yaml
paths:
  data_dir: YOUR_DATA_DIR
```

#### Text decoder

Download [Qwen3-ASR-1.7B](https://huggingface.co/Qwen/Qwen3-ASR-1.7B) to `weights/Qwen3-ASR-1.7B`, the default
`paths.text_decoder_dir`. MAS-BIT uses its language model and tokenizer; the audio tower is not loaded. The revision
below is the one the released results were obtained with.

```bash
hf download Qwen/Qwen3-ASR-1.7B --revision 7278e1e70fe206f11671096ffdd38061171dd6e5 --local-dir weights/Qwen3-ASR-1.7B
```

#### Default paths

Run every command from the repository root: relative paths in the config and in the command-line defaults are resolved
against it. Apart from the data folder, nothing has to be configured if the weights are placed as follows:

```text
Multi_Agent_System_for_BraIn_to_Text/
├── configs/mas_bit.yaml
├── weights/
│   ├── Qwen3-ASR-1.7B/          text decoder and tokenizer        paths.text_decoder_dir    (training and inference)
│   ├── mas_bit.safetensors      released MAS-BIT checkpoint       eval.py --checkpoint      (inference)
│   └── init.safetensors         initial weights from BIT          paths.init_checkpoint     (training)
└── outputs/
    ├── mas_bit/                 checkpoints and logs of train.py  paths.output_dir
    ├── holdout/                 eval.py --eval_split holdout      eval.py --output_dir
    └── val/                     eval.py --eval_split val          eval.py --output_dir
```

`weights/` and `outputs/` are ignored by git. To keep a file elsewhere, point its entry at it, in the config, with
`--kwargs` (e.g. `--kwargs paths.text_decoder_dir=/models/Qwen3-ASR-1.7B`) or with the `eval.py` option.

## Training

MAS-BIT starts from the neural encoders and phoneme readouts of the released [BIT](https://github.com/yzhang511/bit)
checkpoints of T12 and T15. They are provided as one file on [Hugging Face](https://huggingface.co/XinjieZhang/MAS-BIT);
download it to `weights/init.safetensors`, the default `paths.init_checkpoint`:

```bash
hf download XinjieZhang/MAS-BIT init.safetensors --local-dir weights
```

The same file can be extracted from the BIT checkpoints themselves:

```bash
python prepare_init.py --t12_checkpoint YOUR_BIT_T12_CHECKPOINT/model.safetensors \
                       --t15_checkpoint YOUR_BIT_T15_CHECKPOINT/model.safetensors
```

Train on 8 GPUs:

```bash
torchrun --nproc_per_node 8 train.py [--config configs/mas_bit.yaml] [--resume] [--kwargs KEY=VALUE ...]
```
<small>

- *--resume: continue from the latest epoch in `paths.output_dir` (default `outputs/mas_bit`)*
- *--kwargs: override any entry of the config, e.g. `seed=1 paths.output_dir=outputs/seed1`*

</small>

`configs/mas_bit.yaml` controls the model, the augmentation, the optimizers, checkpoint selection and the
staged-to-joint schedule (`training.stages`): the new interfaces are first trained with frozen encoders (`align`,
5 epochs), then together with the encoders (`encoder`, 5), then each agent's adapter in turn (`neural`, `phonetic`,
`arbiter`, 5 epochs each; the phoneme readouts are tuned in the `phonetic` stage), and finally everything jointly
(`joint`, 40 epochs). An update uses `GPUs x batch_size x grad_accum = 8 x 2 x 2 = 32` trials; with fewer GPUs, raise
`training.grad_accum` to keep this product (e.g. `torchrun --nproc_per_node 2 train.py --kwargs training.grad_accum=8`).

After every epoch the pooled validation sets of both participants are decoded greedily. `paths.output_dir` then holds
`best.safetensors` (the checkpoint with the fewest arbiter word errors, from epoch 25 on), `last.safetensors` and
`training_state.pt` (the latest epoch, for `--resume`), the run's `config.yaml` and `history.jsonl` (losses and
validation WER of every epoch).

## Inference with released weights

The released checkpoint is one file on [Hugging Face](https://huggingface.co/XinjieZhang/MAS-BIT); the architecture is
read from `configs/mas_bit.yaml`. Once the setup is done (environment, `paths.data_dir`, `weights/Qwen3-ASR-1.7B`), two
commands decode the hidden competition holdouts of T12 and T15. The first downloads the checkpoint to
`weights/mas_bit.safetensors`, where `eval.py` reads it by default; the second writes `t12_holdout.txt` and
`t15_holdout.csv` to `outputs/holdout/`:

```bash
hf download XinjieZhang/MAS-BIT mas_bit.safetensors --local-dir weights
python eval.py --eval_split holdout
```

Score the provided validation sets:

```bash
python eval.py --eval_split val
```
<small>

- *--checkpoint: default `weights/mas_bit.safetensors`; a checkpoint written by `train.py`, e.g.
  `outputs/mas_bit/best.safetensors`, is used the same way*
- *--eval_split: holdout, val*
- *--participants: t12, t15 (default: both)*
- *--output_dir: default `outputs/EVAL_SPLIT`*
- *`holdout` writes `t12_holdout.txt` (one sentence per line) and `t15_holdout.csv` (`id,text`), the formats of the
  Brain-to-Text '24 and '25 benchmarks*
- *`val` prints the word error rate of the two experts and the arbiter of each participant*

</small>

Decoding is greedy, in single precision, one trial at a time; on one NVIDIA A40 the T12 and T15 holdouts take about 27
and 35 minutes. The arbiter's sentence is the output of the system; the two expert drafts are saved next to it in
`predictions.jsonl`. Uploaded to the benchmarks, the holdout files of the released checkpoint score a holdout WER of
6.929% on Brain-to-Text '24 (T12) and 3.606% (public) / 3.892% (private) on Brain-to-Text '25 (T15).

## Code structure

```text
configs/mas_bit.yaml     model, augmentation, training schedule and optimizers
data/                    loaders of the two benchmarks
models/encoder.py        BIT neural encoder with participant-specific input projections
models/text_decoder.py   Qwen3-ASR text decoder with one LoRA adapter per agent
models/mas_bit.py        neural and phonetic experts, latent messages, arbiter, training objective
models/augmentation.py   neural input augmentation
utils/                   batching, sampler, staged schedule, optimizers, evaluation, checkpoints
prepare_init.py          initial weights from the released BIT checkpoints (weights/init.safetensors)
train.py                 joint training on T12 and T15
eval.py                  validation scoring and holdout decoding
```

## Acknowledgements

The neural encoder and the data pipeline build on [BIT](https://github.com/yzhang511/bit).
