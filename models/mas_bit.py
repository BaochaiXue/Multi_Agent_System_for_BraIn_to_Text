"""MAS-BIT: a multi-agent system for brain-to-text decoding.

    recording of participant s
        -> encoder_t12 (input projection of s) -> H_t12, phoneme posteriors Q_t12
        -> encoder_t15 (input projection of s) -> H_t15, phoneme posteriors Q_t15
    Z_N = fusion([H_t12; H_t15])                 neural view
    Z_P = [Q_t12; Q_t15] @ phoneme embeddings    phonetic view

    neural expert    reads Z_N                -> draft, latent message M_N
    phonetic expert  reads Z_P                -> draft, latent message M_P
    arbiter          reads [Z_N; M_N; M_P]    -> sentence

The three agents are LoRA adapters on one frozen text decoder.  A latent message is the sequence of
probability-weighted token embeddings (top-k) obtained by replaying an expert's own draft through that expert, so the
arbiter loss reaches the experts, the interfaces and the encoders.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

from models.augmentation import NeuralAugmentation, span_patch_mask
from models.encoder import NeuralEncoder
from models.text_decoder import AGENTS, add_agent_adapters, load_text_decoder

PARTICIPANTS = ("t12", "t15")
#: input channels per participant (threshold crossings and spike-band power of every electrode)
CHANNELS = {"t12": 256, "t15": 512}
EXPERTS = ("neural", "phonetic")
N_PHONEMES = 41
BLANK_ID = 0
EOS_IDS = (151643, 151645)
IGNORE = -100
SLOT = "#"

#: parameter groups that the staged schedule switches on and off, by parameter-name prefix (first match wins)
PARAMETER_GROUPS = (
    ("encoders.t12.input_projections.t12.", "input_projection_native"),
    ("encoders.t15.input_projections.t15.", "input_projection_native"),
    ("encoders.t12.input_projections.t15.", "input_projection_cross"),
    ("encoders.t15.input_projections.t12.", "input_projection_cross"),
    ("encoders.", "encoder"),
    ("phoneme_heads.", "phoneme_heads"),
    ("neural_projector.", "neural_projector"),
    ("phoneme_embeddings.", "phoneme_embeddings"),
    ("role_embeddings.", "role_embeddings"),
)


def parameter_group(name: str) -> str | None:
    """Group of an adaptation parameter; None for the frozen text decoder."""
    if "lora_" in name:
        return "adapter_" + next(agent for agent in AGENTS if f".{agent}." in name)
    return next((group for prefix, group in PARAMETER_GROUPS if name.startswith(prefix)), None)


class FusionProjector(nn.Module):
    """[LN(H_t12); LN(H_t15)] -> Linear -> GELU -> Linear: the neural view in the text decoder's embedding space."""

    def __init__(self, encoder_size: int, hidden_size: int, decoder_size: int):
        super().__init__()
        self.norms = nn.ModuleDict({participant: nn.LayerNorm(encoder_size) for participant in PARTICIPANTS})
        self.fc1 = nn.Linear(len(PARTICIPANTS) * encoder_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, decoder_size)

    def forward(self, hidden: dict[str, torch.Tensor]) -> torch.Tensor:
        x = torch.cat([self.norms[participant](hidden[participant]) for participant in PARTICIPANTS], dim=-1)
        return self.fc2(F.gelu(self.fc1(x)))


@dataclass
class AgentPass:
    """Teacher-forced pass of one agent over the answer positions (plus the position before the first answer token)."""

    logits: torch.Tensor
    labels: torch.Tensor


@dataclass
class Losses:
    """Sums over one batch; the trainer normalises them by the tokens and phonemes of the whole optimizer update."""

    cross_entropy: dict[str, torch.Tensor]
    ctc: torch.Tensor
    intermediate_ctc: torch.Tensor
    text_tokens: torch.Tensor
    phonemes: torch.Tensor


class MASBIT(nn.Module):
    """The text decoder is held in bfloat16; the encoders, interfaces and adapters in float32."""

    def __init__(self, config, tokenizer):
        super().__init__()
        model = config.model
        self.encoders = nn.ModuleDict(
            {participant: NeuralEncoder(config.encoder, CHANNELS) for participant in PARTICIPANTS}
        )
        encoder_size = config.encoder.hidden_size
        self.phoneme_heads = nn.ModuleDict(
            {participant: nn.Linear(encoder_size, N_PHONEMES) for participant in PARTICIPANTS}
        )
        self.text_decoder = add_agent_adapters(
            load_text_decoder(config.paths.text_decoder_dir), model.lora_rank, model.lora_alpha, model.lora_dropout
        )
        embeddings = self.text_decoder.get_input_embeddings().weight
        decoder_size = embeddings.shape[1]
        self.neural_projector = FusionProjector(encoder_size, model.fusion_hidden_size, decoder_size)
        # one block of phoneme embeddings per encoder; the posteriors of each encoder sum to one
        self.phoneme_embeddings = nn.Embedding(len(PARTICIPANTS) * N_PHONEMES, decoder_size)
        nn.init.normal_(
            self.phoneme_embeddings.weight, std=float(embeddings.detach().float().std()) / len(PARTICIPANTS) ** 0.5
        )
        self.role_embeddings = nn.Embedding(len(EXPERTS), decoder_size)
        nn.init.zeros_(self.role_embeddings.weight)

        self.prompt_splits = {}
        for agent in AGENTS:
            before, after = (
                tokenizer(part, add_special_tokens=False)["input_ids"] for part in model.prompts[agent].split(SLOT)
            )
            self.register_buffer(f"prompt_{agent}", torch.tensor(before + after), persistent=False)
            self.prompt_splits[agent] = len(before)

        self.message_top_k = model.message_top_k
        self.message_dropout, self.message_token_dropout = model.message_dropout, model.message_token_dropout
        self.max_new_tokens = dict(model.max_new_tokens)
        self.span_mask_frac = config.augmentation.span_mask_frac
        self.intermediate_ctc_layers = tuple(model.intermediate_ctc_layers)
        self.augmentation = NeuralAugmentation(
            config.augmentation, config.encoder.patch_size, config.encoder.max_frames
        )
        self.ctc_loss = nn.CTCLoss(blank=BLANK_ID, reduction="sum", zero_infinity=True)
        self._adapter_parameters = [(name, p) for name, p in self.named_parameters() if "lora_" in name]
        self.set_trainable(None)

    # ------------------------------------------------------------------ trainable parameters
    def adaptation_parameters(self):
        """(name, parameter) of everything MAS-BIT trains: encoders, interfaces, phoneme readouts and the adapters."""
        return [(name, p) for name, p in self.named_parameters() if parameter_group(name) is not None]

    def set_trainable(self, groups):
        """Restrict gradients to the parameter groups of the current training stage (None: all groups)."""
        self._trainable = {
            name for name, _ in self.adaptation_parameters() if groups is None or parameter_group(name) in groups
        }
        for name, parameter in self.named_parameters():
            parameter.requires_grad_(name in self._trainable)

    def activate(self, agent: str):
        self.text_decoder.set_adapter(agent)
        # selecting an adapter resets the gradient flags of all adapters; restore the stage's flags
        for name, parameter in self._adapter_parameters:
            parameter.requires_grad_(name in self._trainable)

    def adaptation_state_dict(self) -> dict[str, torch.Tensor]:
        """The weights MAS-BIT adds to the frozen text decoder (what a checkpoint stores)."""
        state = {name: p.detach().float().cpu() for name, p in self.adaptation_parameters()}
        state.update(
            {
                f"encoders.{participant}.mask_token": self.encoders[participant].mask_token.detach().float().cpu()
                for participant in PARTICIPANTS
            }
        )
        return state

    def load_adaptation_state_dict(self, state: dict[str, torch.Tensor]):
        expected = set(self.adaptation_state_dict())
        if set(state) != expected:
            raise ValueError(
                f"Checkpoint differs from the model: missing {sorted(expected - set(state))[:4]}, "
                f"unexpected {sorted(set(state) - expected)[:4]}"
            )
        self.load_state_dict(state, strict=False)

    # ------------------------------------------------------------------ neural and phonetic views
    def encode(self, spikes, spikes_mask, spikes_timestamp, participant, patch_mask=None, tap_layers=()):
        hidden, tapped, mask = {}, {}, None
        for origin in PARTICIPANTS:
            hidden[origin], mask, tapped[origin] = self.encoders[origin](
                spikes, spikes_mask, spikes_timestamp, participant, patch_mask=patch_mask, tap_layers=tap_layers
            )
        return hidden, mask, tapped

    def phoneme_log_probs(self, origin: str, hidden: torch.Tensor) -> torch.Tensor:
        return F.log_softmax(self.phoneme_heads[origin](hidden.float()), dim=-1)

    def views(self, hidden: dict[str, torch.Tensor]):
        """(Z_N, Z_P, phoneme log-posteriors of each encoder)."""
        log_probs = {origin: self.phoneme_log_probs(origin, hidden[origin]) for origin in PARTICIPANTS}
        posteriors = torch.cat([log_probs[origin].exp() for origin in PARTICIPANTS], dim=-1)
        phonetic = posteriors.to(self.phoneme_embeddings.weight.dtype) @ self.phoneme_embeddings.weight
        return self.neural_projector(hidden), phonetic, log_probs

    # ------------------------------------------------------------------ text decoder
    def prompt(self, agent: str, batch: int):
        ids = getattr(self, f"prompt_{agent}").unsqueeze(0).expand(batch, -1)
        return ids, torch.ones_like(ids), self.prompt_splits[agent]

    @staticmethod
    def insert_prefix(text_embeds, attention_mask, labels, prefix, prefix_mask, split: int):
        """Place the continuous prefix in the prompt's slot: [prompt before the slot; prefix; prompt after; answer]."""
        merged = torch.cat((text_embeds[:, :split], prefix, text_embeds[:, split:]), dim=1)
        mask = torch.cat((attention_mask[:, :split], prefix_mask, attention_mask[:, split:]), dim=1)
        if labels is not None:
            labels = torch.cat((labels[:, :split], torch.full_like(prefix_mask, IGNORE), labels[:, split:]), dim=1)
        return merged, mask, labels

    def embed(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.text_decoder.get_input_embeddings()(input_ids)

    def teacher_forced(self, agent, input_ids, attention_mask, labels, prefix, prefix_mask) -> AgentPass:
        self.activate(agent)
        text_embeds = self.embed(input_ids)
        merged, mask, labels = self.insert_prefix(
            text_embeds,
            attention_mask,
            labels,
            prefix.to(text_embeds.dtype),
            prefix_mask.long(),
            self.prompt_splits[agent],
        )
        positions = mask.cumsum(-1) - 1
        positions.masked_fill_(mask == 0, 1)
        keep = min(labels.shape[1], int((labels != IGNORE).sum(dim=1).max()) + 1)
        logits = self.text_decoder(
            inputs_embeds=merged.to(self.text_decoder.dtype),
            attention_mask=mask,
            position_ids=positions,
            use_cache=False,
            logits_to_keep=keep,
        ).logits
        return AgentPass(logits, labels[:, -logits.shape[1] :])

    def greedy(self, agent, prefix, prefix_mask, max_new_tokens: int) -> torch.Tensor:
        self.activate(agent)
        ids, attention_mask, split = self.prompt(agent, prefix.shape[0])
        text_embeds = self.embed(ids)
        merged, mask, _ = self.insert_prefix(
            text_embeds, attention_mask, None, prefix.to(text_embeds.dtype), prefix_mask.long(), split
        )
        return self.text_decoder.generate(
            inputs_embeds=merged.to(self.text_decoder.dtype),
            attention_mask=mask,
            max_new_tokens=max_new_tokens,
            num_beams=1,
            do_sample=False,
            pad_token_id=EOS_IDS[0],
            eos_token_id=list(EOS_IDS),
            use_cache=True,
        )

    def answer_inputs(self, agent: str, answers: list[torch.Tensor]):
        """[prompt; answer] token ids of every row, right padded, with labels on the answer positions only."""
        prompt = getattr(self, f"prompt_{agent}")
        ids = [torch.cat((prompt, answer)) for answer in answers]
        labels = [torch.cat((torch.full_like(prompt, IGNORE), answer)) for answer in answers]
        return (
            pad_sequence(ids, batch_first=True, padding_value=EOS_IDS[0]),
            pad_sequence([torch.ones_like(row) for row in ids], batch_first=True),
            pad_sequence(labels, batch_first=True, padding_value=IGNORE),
        )

    # ------------------------------------------------------------------ latent messages
    @staticmethod
    def trim(generated: torch.Tensor) -> list[torch.Tensor]:
        """Every draft up to and including its first end-of-sequence token."""
        is_eos = torch.zeros_like(generated, dtype=torch.bool)
        for eos in EOS_IDS:
            is_eos |= generated == eos
        ends = torch.where(is_eos.any(dim=1), is_eos.int().argmax(dim=1) + 1, generated.shape[1]).tolist()
        return [row[:end] if end else row.new_tensor([EOS_IDS[0]]) for row, end in zip(generated, ends)]

    def latent_message(self, expert: str, draft: torch.Tensor, prefix, prefix_mask):
        """Replay ``draft`` through its expert and return the probability-weighted token embeddings of every draft
        position (top-k of the expert's next-token distribution), plus the expert's role embedding."""
        input_ids, attention_mask, labels = self.answer_inputs(expert, self.trim(draft))
        replay = self.teacher_forced(expert, input_ids, attention_mask, labels, prefix, prefix_mask)
        logits, active = replay.logits[:, :-1], replay.labels[:, 1:] != IGNORE
        top_logits, top_ids = logits.float().topk(self.message_top_k, dim=-1)
        weights = torch.softmax(top_logits, dim=-1)
        embeddings = self.text_decoder.get_input_embeddings().weight
        soft = (weights.unsqueeze(-1).to(embeddings.dtype) * embeddings[top_ids]).sum(dim=-2)
        # left-align the active positions of every row
        lengths = active.sum(dim=1)
        slot = (active.long().cumsum(dim=1) - 1).clamp_min(0)
        message = soft.new_zeros(soft.shape[0], max(int(lengths.max()), 1), soft.shape[-1])
        message.scatter_add_(
            1, slot.unsqueeze(-1).expand_as(soft), torch.where(active.unsqueeze(-1), soft, torch.zeros_like(soft))
        )
        mask = (torch.arange(message.shape[1], device=soft.device).unsqueeze(0) < lengths.unsqueeze(1)).long()
        if self.training and self.message_dropout > 0.0:
            keep = torch.rand(soft.shape[0], 1, 1, device=soft.device) >= self.message_dropout
            message = message * keep.to(message.dtype)
        message = F.dropout(message, p=self.message_token_dropout, training=self.training)
        return message + self.role_embeddings.weight[EXPERTS.index(expert)].to(message.dtype), mask

    def deliberate(self, neural, phonetic, mask, max_new_tokens: int):
        """Expert drafts (no gradient) and the arbiter prefix [Z_N; M_N; M_P] built from their replayed messages."""
        views = {"neural": neural, "phonetic": phonetic}
        was_training = self.text_decoder.training
        self.text_decoder.eval()
        with torch.no_grad():
            drafts = {expert: self.greedy(expert, views[expert], mask, max_new_tokens) for expert in EXPERTS}
        self.text_decoder.train(was_training)
        messages = [self.latent_message(expert, drafts[expert].detach(), views[expert], mask) for expert in EXPERTS]
        prefix = torch.cat([neural] + [message.to(neural.dtype) for message, _ in messages], dim=1)
        prefix_mask = torch.cat([mask.long()] + [message_mask for _, message_mask in messages], dim=1)
        return drafts, prefix, prefix_mask

    # ------------------------------------------------------------------ training objective
    def ctc(self, log_probs: torch.Tensor, mask: torch.Tensor, phonemes: torch.Tensor) -> torch.Tensor:
        valid = phonemes >= 0
        return self.ctc_loss(
            log_probs.transpose(0, 1), phonemes[valid].long(), mask.sum(dim=1).long(), valid.sum(dim=1).long()
        )

    def forward(self, spikes, spikes_mask, spikes_timestamp, participant: str, targets, phonemes) -> Losses:
        """``targets``: answer token ids (sentence and end of sequence) padded with -100;
        ``phonemes``: CTC targets padded with -1."""
        patch_mask, tap_layers = None, ()
        if self.training:
            spikes, spikes_mask, spikes_timestamp = self.augmentation(spikes, spikes_mask, spikes_timestamp, phonemes)
            patch_mask = span_patch_mask(spikes_mask, self.encoders[PARTICIPANTS[0]].patch_size, self.span_mask_frac)
            tap_layers = self.intermediate_ctc_layers
        hidden, mask, tapped = self.encode(spikes, spikes_mask, spikes_timestamp, participant, patch_mask, tap_layers)
        neural, phonetic, log_probs = self.views(hidden)
        _, arbiter_prefix, arbiter_mask = self.deliberate(neural, phonetic, mask, self.max_new_tokens[participant])

        answers = [row[row != IGNORE] for row in targets]
        cross_entropy = {}
        for agent, prefix, prefix_mask in (
            ("neural", neural, mask),
            ("phonetic", phonetic, mask),
            ("arbiter", arbiter_prefix, arbiter_mask),
        ):
            agent_pass = self.teacher_forced(agent, *self.answer_inputs(agent, answers), prefix, prefix_mask)
            cross_entropy[agent] = F.cross_entropy(
                agent_pass.logits[:, :-1].reshape(-1, agent_pass.logits.shape[-1]).float(),
                agent_pass.labels[:, 1:].reshape(-1),
                ignore_index=IGNORE,
                reduction="sum",
            )
        ctc = torch.stack([self.ctc(log_probs[origin], mask, phonemes) for origin in PARTICIPANTS]).mean()
        intermediate = [
            self.ctc(self.phoneme_log_probs(origin, state), mask, phonemes)
            for origin in PARTICIPANTS
            for state in tapped[origin]
        ]
        intermediate_ctc = torch.stack(intermediate).mean() if intermediate else ctc.new_zeros(())
        return Losses(cross_entropy, ctc, intermediate_ctc, (targets != IGNORE).sum(), (phonemes >= 0).sum())

    # ------------------------------------------------------------------ inference
    @torch.inference_mode()
    def generate(self, spikes, spikes_mask, spikes_timestamp, participant: str) -> dict[str, torch.Tensor]:
        """Greedy token ids of the two expert drafts and of the arbiter's sentence (``"arbiter"`` is the output)."""
        hidden, mask, _ = self.encode(spikes, spikes_mask, spikes_timestamp, participant)
        neural, phonetic, _ = self.views(hidden)
        max_new_tokens = self.max_new_tokens[participant]
        drafts, prefix, prefix_mask = self.deliberate(neural, phonetic, mask, max_new_tokens)
        return {**drafts, "arbiter": self.greedy("arbiter", prefix, prefix_mask, max_new_tokens)}
