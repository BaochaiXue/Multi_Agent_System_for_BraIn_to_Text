"""Neural encoder of BIT (Zhang et al., 2026) with participant-specific input projections.

A recording is smoothed over time, cut into non-overlapping patches of ``patch_size`` time bins, projected to the model
width by the input projection of the recorded participant, and processed by bidirectional Transformer blocks with
rotary position embeddings.  The released BIT encoders load into this module unchanged (see ``prepare_init.py``).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import signal

PAD_VALUE = -100.0


def rotary_tables(head_size: int, max_frames: int, base: float) -> tuple[torch.Tensor, torch.Tensor]:
    inv_freq = 1.0 / (base ** (torch.arange(0, head_size, 2).float() / head_size))
    freqs = torch.einsum("i,j->ij", torch.arange(max_frames, dtype=inv_freq.dtype), inv_freq)
    angles = torch.cat((freqs, freqs), dim=-1)
    return angles.cos(), angles.sin()


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


class GaussianSmoothing(nn.Module):
    """Gaussian smoothing of every channel along time."""

    def __init__(self, sd: int):
        super().__init__()
        kernel = torch.from_numpy(signal.windows.gaussian(1 + 6 * sd, sd))
        self.register_buffer("kernel", kernel / kernel.sum(), persistent=False)

    def forward(self, spikes: torch.Tensor) -> torch.Tensor:
        channels = spikes.shape[-1]
        kernel = self.kernel[None, None].expand(channels, 1, -1).to(spikes.dtype)
        return F.conv1d(spikes.transpose(1, 2), kernel, padding="same", groups=channels).transpose(1, 2)


class Attention(nn.Module):
    def __init__(self, hidden_size: int, n_heads: int, dropout: float, max_frames: int, rope_theta: float):
        super().__init__()
        self.n_heads, self.head_size, self.attn_dropout = n_heads, hidden_size // n_heads, dropout
        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
        cos, sin = rotary_tables(self.head_size, max_frames, rope_theta)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    def split_heads(self, x: torch.Tensor) -> torch.Tensor:
        """(B, S, width) -> (B, heads, S, head_size)."""
        batch, tokens, _ = x.shape
        return x.reshape(batch, tokens, self.n_heads, self.head_size).permute(0, 2, 1, 3)

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor, timestamp: torch.Tensor) -> torch.Tensor:
        batch, tokens, width = x.shape
        q, k, v = self.split_heads(self.query(x)), self.split_heads(self.key(x)), self.split_heads(self.value(x))
        cos, sin = self.cos[timestamp].unsqueeze(1), self.sin[timestamp].unsqueeze(1)
        q = q * cos + rotate_half(q) * sin
        k = k * cos + rotate_half(k) * sin
        mask = attn_mask.unsqueeze(1).expand(batch, self.n_heads, tokens, tokens).bool()
        out = F.scaled_dot_product_attention(
            q.to(v.dtype), k.to(v.dtype), v, attn_mask=mask, dropout_p=self.attn_dropout if self.training else 0.0
        )
        out = out.permute(0, 2, 1, 3).reshape(batch, tokens, width).to(x.dtype)
        return self.out_proj(self.dropout(out))


class MLP(nn.Module):
    def __init__(self, hidden_size: int, inter_size: int, dropout: float):
        super().__init__()
        self.up_proj = nn.Linear(hidden_size, inter_size)
        self.down_proj = nn.Linear(inter_size, hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.down_proj(F.gelu(self.up_proj(x))))


class EncoderLayer(nn.Module):
    def __init__(
        self, hidden_size: int, n_heads: int, inter_size: int, dropout: float, max_frames: int, rope_theta: float
    ):
        super().__init__()
        self.ln1 = nn.LayerNorm(hidden_size)
        self.attn = Attention(hidden_size, n_heads, dropout, max_frames, rope_theta)
        self.ln2 = nn.LayerNorm(hidden_size)
        self.mlp = MLP(hidden_size, inter_size, dropout)

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor, timestamp: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x), attn_mask, timestamp)
        return x + self.mlp(self.ln2(x))


class NeuralEncoder(nn.Module):
    """``config``: the ``encoder`` block of ``configs/mas_bit.yaml``.
    ``channels``: number of input channels of each participant."""

    def __init__(self, config, channels: dict[str, int]):
        super().__init__()
        self.hidden_size, self.patch_size = config.hidden_size, config.patch_size
        self.smoothing = GaussianSmoothing(config.smooth_sd)
        self.input_projections = nn.ModuleDict(
            {
                participant: nn.Sequential(
                    nn.LayerNorm(count * self.patch_size),
                    nn.Linear(count * self.patch_size, self.hidden_size),
                    nn.LayerNorm(self.hidden_size),
                )
                for participant, count in channels.items()
            }
        )
        self.input_dropout = nn.Dropout(config.input_dropout)
        # embedding that replaces span-masked patches during training; fixed, taken from the released encoder
        self.register_buffer("mask_token", torch.zeros(1, 1, self.hidden_size))
        self.layers = nn.ModuleList(
            [
                EncoderLayer(
                    self.hidden_size,
                    config.n_heads,
                    config.inter_size,
                    config.dropout,
                    config.max_frames,
                    config.rope_theta,
                )
                for _ in range(config.n_layers)
            ]
        )
        self.out_norm = nn.LayerNorm(self.hidden_size)

    def patch(self, spikes: torch.Tensor, mask: torch.Tensor, timestamp: torch.Tensor):
        """(B, T, C) time bins -> (B, T / patch, patch * C) patches; a patch is valid if any of its bins is."""
        batch, frames, channels = spikes.shape
        pad = (self.patch_size - frames % self.patch_size) % self.patch_size
        if pad:
            spikes = F.pad(spikes, (0, 0, 0, pad), value=PAD_VALUE)
            mask, timestamp = F.pad(mask, (0, pad)), F.pad(timestamp, (0, pad))
        patches = spikes.shape[1] // self.patch_size
        spikes = spikes.reshape(batch, patches, self.patch_size * channels)
        mask = mask.reshape(batch, patches, self.patch_size).any(dim=2).long()
        return spikes, mask, timestamp[:, :: self.patch_size]

    def forward(self, spikes, spikes_mask, spikes_timestamp, participant: str, patch_mask=None, tap_layers=()):
        """Returns the encoded tokens (B, S, hidden), their validity mask (B, S) and the states after ``tap_layers``
        (each passed through the output norm), used by the intermediate CTC losses."""
        x, mask, timestamp = self.patch(self.smoothing(spikes), spikes_mask, spikes_timestamp)
        x = self.input_projections[participant](x)
        if patch_mask is not None:
            x = torch.where(patch_mask.unsqueeze(-1), self.mask_token.expand_as(x).to(x.dtype), x)
        x = self.input_dropout(x)
        tokens = x.shape[1]
        eye = torch.eye(tokens, device=x.device, dtype=torch.int64).expand(x.shape[0], tokens, tokens)
        attn_mask = eye | mask.unsqueeze(1).expand(-1, tokens, -1)
        tapped = {}
        for index, layer in enumerate(self.layers, start=1):
            x = layer(x, attn_mask, timestamp)
            if index in tap_layers:
                tapped[index] = self.out_norm(x)
        return self.out_norm(x), mask, [tapped[index] for index in tap_layers]
