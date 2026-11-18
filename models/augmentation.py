"""Neural input augmentation, applied once per training example before the encoder so that all three agents read the
same augmented recording.

With probability ``apply_prob`` per example: a local time warp, electrode dropout, short time masks and white noise
on the raw (z-scored) features.
Columns hold [threshold crossings | spike-band power], so electrode ``e`` owns columns ``e`` and ``E + e``.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def ctc_min_tokens(phonemes: torch.Tensor) -> int:
    """Shortest input a CTC alignment of ``phonemes`` needs: one token per label plus one blank per repeated pair."""
    if phonemes.numel() == 0:
        return 0
    return int(phonemes.numel()) + int((phonemes[1:] == phonemes[:-1]).sum())


class NeuralAugmentation:
    def __init__(self, config, patch_size: int, max_frames: int):
        self.apply_prob = config.apply_prob
        self.warp_points, self.warp_sd, self.warp_clip = config.warp_points, config.warp_sd, tuple(config.warp_clip)
        self.electrode_dropout, self.noise_sd = config.electrode_dropout, config.noise_sd
        self.time_mask_frac, self.time_mask_span = config.time_mask_frac, tuple(config.time_mask_span)
        self.patch_size, self.max_frames = patch_size, max_frames

    def warp(self, x: torch.Tensor, min_tokens: int):
        """Resample ``warp_points + 1`` segments with factors drawn around one; None if the result would be too long
        for the encoder or too short for the CTC alignment of the example's phonemes."""
        frames = x.shape[0]
        if frames < 3 * (self.warp_points + 1):
            return None
        cuts = torch.sort(torch.randint(1, frames - 1, (self.warp_points,)).unique()).values.tolist()
        bounds = [0] + cuts + [frames]
        segments = []
        for start, stop in zip(bounds[:-1], bounds[1:]):
            factor = float(torch.randn(()).item() * self.warp_sd + 1.0)
            factor = min(max(factor, self.warp_clip[0]), self.warp_clip[1])
            segments.append((start, stop, max(1, int(round((stop - start) * factor)))))
        total = sum(size for _, _, size in segments)
        if total > self.max_frames or math.ceil(total / self.patch_size) < min_tokens:
            return None
        pieces = []
        for start, stop, size in segments:
            segment = x[start:stop].transpose(0, 1).unsqueeze(0).float()
            if stop - start > 1:
                segment = F.interpolate(segment, size=size, mode="linear", align_corners=size > 1)
            else:
                segment = segment.expand(1, segment.shape[1], size)
            pieces.append(segment.squeeze(0).transpose(0, 1).to(x.dtype))
        return torch.cat(pieces, dim=0)

    @torch.no_grad()
    def __call__(self, spikes, spikes_mask, spikes_timestamp, phonemes):
        """Returns the augmented batch (zero padded), its mask and its time stamps; lengths follow the time warp."""
        batch, _, channels = spikes.shape
        electrodes = channels // 2
        lengths = spikes_mask.long().sum(1).tolist()
        apply = (torch.rand(batch) < self.apply_prob).tolist()
        rows = []
        for b in range(batch):
            x = spikes[b, : lengths[b]]
            if not apply[b]:
                rows.append(x)
                continue
            warped = self.warp(x, ctc_min_tokens(phonemes[b][phonemes[b] >= 0]))
            x = (x if warped is None else warped).clone()
            dropped = torch.nonzero(torch.rand(electrodes, device=x.device) < self.electrode_dropout).flatten()
            if dropped.numel():
                x[:, dropped] = 0.0
                x[:, dropped + electrodes] = 0.0
            budget, used = int(self.time_mask_frac * x.shape[0]), 0
            while used < budget:
                span = min(int(torch.randint(self.time_mask_span[0], self.time_mask_span[1] + 1, (1,))), budget - used)
                if span < 1 or span > x.shape[0]:
                    break
                start = int(torch.randint(0, x.shape[0] - span + 1, (1,)))
                x[start : start + span] = 0.0
                used += span
            if self.noise_sd > 0:
                x = x + self.noise_sd * torch.randn_like(x)
            rows.append(x)
        frames = max(x.shape[0] for x in rows)
        out = spikes.new_zeros(batch, frames, channels)
        mask = torch.zeros(batch, frames, dtype=spikes_mask.dtype, device=spikes_mask.device)
        timestamp = torch.zeros(batch, frames, dtype=spikes_timestamp.dtype, device=spikes_timestamp.device)
        for b, x in enumerate(rows):
            out[b, : x.shape[0]], mask[b, : x.shape[0]] = x, 1
            timestamp[b, : x.shape[0]] = torch.arange(x.shape[0], device=timestamp.device, dtype=timestamp.dtype)
        return out, mask, timestamp


@torch.no_grad()
def span_patch_mask(spikes_mask: torch.Tensor, patch_size: int, max_frac: float) -> torch.Tensor:
    """One contiguous span of patches per example, at most ``max_frac`` of its valid patches, to be replaced by the
    encoder's mask embedding.  Returns a (B, patches) boolean mask on the padded patch grid."""
    batch, frames = spikes_mask.shape
    grid = (frames + patch_size - 1) // patch_size
    valid = ((spikes_mask.long().sum(1) + patch_size - 1) // patch_size).clamp(max=grid)
    longest = (valid.float() * max_frac).floor().long()
    lengths = torch.where(
        longest >= 1,
        (torch.rand(batch, device=valid.device) * longest.float()).floor().long() + 1,
        torch.zeros_like(longest),
    )
    starts = (torch.rand(batch, device=valid.device) * (valid - lengths + 1).clamp(min=1).float()).floor().long()
    index = torch.arange(grid, device=valid.device).unsqueeze(0).expand(batch, grid)
    return (index >= starts.unsqueeze(1)) & (index < (starts + lengths).unsqueeze(1))
