"""Text decoder shared by the three agents: the language model of Qwen3-ASR with one LoRA adapter per agent.

Qwen3-ASR stores its causal language model under ``thinker.*`` next to an audio tower.  MAS-BIT feeds the decoder
with neural embeddings, so only the language model is loaded; the audio tower is never materialised.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model
from safetensors import safe_open
from transformers import Qwen2Tokenizer, Qwen3Config, Qwen3ForCausalLM

AGENTS = ("neural", "phonetic", "arbiter")
LORA_TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


def load_tokenizer(model_dir: str | Path) -> Qwen2Tokenizer:
    """Load the tokenizer of a local copy of ``Qwen/Qwen3-ASR-1.7B``.

    It is a Qwen2 tokenizer.  Loading that class directly skips the multimodal ``config.json``, which ``AutoTokenizer``
    would read and warn about.
    """
    return Qwen2Tokenizer.from_pretrained(str(model_dir), add_bos_token=False, add_eos_token=False)


def load_text_decoder(model_dir: str | Path, dtype: torch.dtype = torch.bfloat16) -> Qwen3ForCausalLM:
    """Load ``thinker.model`` and ``thinker.lm_head`` from a local copy of ``Qwen/Qwen3-ASR-1.7B``."""
    model_dir = Path(model_dir)
    config = dict(json.loads((model_dir / "config.json").read_text())["thinker_config"]["text_config"])
    config.pop("model_type", None)
    rope = config.get("rope_scaling")
    if isinstance(rope, dict) and rope.get("rope_type", rope.get("type")) == "default":
        config.pop("rope_scaling")
    config = Qwen3Config(**config)
    config._attn_implementation = "sdpa"
    model = Qwen3ForCausalLM(config).to(dtype=dtype)

    expected, state = set(model.state_dict()), {}
    for shard in sorted(model_dir.glob("*.safetensors")):
        with safe_open(str(shard), framework="pt", device="cpu") as tensors:
            for key in tensors.keys():
                name = key.removeprefix("thinker.")
                if key.startswith("thinker.") and name in expected:
                    state[name] = tensors.get_tensor(key)
    missing, unexpected = model.load_state_dict(state, strict=False)
    missing = [key for key in missing if not key.endswith("rotary_emb.inv_freq")]
    if missing or unexpected:
        raise RuntimeError(
            f"Text decoder weights do not match {model_dir}: missing {missing[:4]}, unexpected {unexpected[:4]}"
        )
    model.tie_weights()
    return model


def add_agent_adapters(model: Qwen3ForCausalLM, rank: int, alpha: int, dropout: float):
    """Freeze the decoder and attach one LoRA adapter per agent; ``set_adapter(agent)`` selects the active one."""
    lora = LoraConfig(
        inference_mode=False, r=rank, lora_alpha=alpha, lora_dropout=dropout, target_modules=list(LORA_TARGETS)
    )
    decoder = get_peft_model(model, lora, adapter_name=AGENTS[0])
    for agent in AGENTS[1:]:
        decoder.add_adapter(agent, lora)
    return decoder
