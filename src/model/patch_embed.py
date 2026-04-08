from __future__ import annotations

import torch
import torch.nn as nn
from einops import rearrange


def _trunc_normal_(tensor: torch.Tensor, std: float = 0.02) -> torch.Tensor:
    return nn.init.trunc_normal_(tensor, std=std)


class ConvFrontend(nn.Module):
    """Two Conv2d blocks with GELU.

    Shapes:
        Input: (batch, 1, n_mels, time)
        Output: (batch, dim, n_mels, time)
    """

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, dim, kernel_size=3, stride=1, padding=1)
        self.act = nn.GELU()
        self.conv2 = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.act(x)
        x = self.conv2(x)
        return x


class PatchEmbedding(nn.Module):
    """Non-overlapping patch linear projection.

    Shapes:
        Input: (batch, dim_in, n_mels, time)
        Output: (batch, num_patches, dim) with num_patches = num_freq_patches * num_time_patches
    """

    def __init__(
        self,
        n_mels: int = 128,
        patch_freq: int = 16,
        patch_time: int = 4,
        dim: int = 256,
        dim_in: int = 256,
    ) -> None:
        super().__init__()
        self.n_mels = n_mels
        self.patch_freq = patch_freq
        self.patch_time = patch_time
        self.dim = dim
        self.dim_in = dim_in
        if n_mels % patch_freq != 0:
            raise ValueError(f"n_mels ({n_mels}) must be divisible by patch_freq ({patch_freq})")
        self.num_freq_patches = n_mels // patch_freq
        patch_dim = patch_freq * patch_time * dim_in
        self.proj = nn.Linear(patch_dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        if h != self.n_mels:
            raise ValueError(f"height {h} != n_mels {self.n_mels}")
        if c != self.dim_in:
            raise ValueError(f"channels {c} != dim_in {self.dim_in}")
        if w % self.patch_time != 0:
            raise ValueError(f"time {w} must be divisible by patch_time {self.patch_time}")
        x = rearrange(
            x,
            "b c (hf pf) (wf pt) -> b (hf wf) (c pf pt)",
            pf=self.patch_freq,
            pt=self.patch_time,
        )
        return self.proj(x)


class PositionalEncoding2D(nn.Module):
    """Learnable 2D patch positions: sum of freq and time tables.

    Shapes:
        Input x: (batch, height * width, dim)
        height, width: patch grid (e.g. num_freq_patches, num_time_patches)
        Output: (batch, height * width, dim)
    """

    def __init__(self, dim: int, max_freq_patches: int, max_time_patches: int) -> None:
        super().__init__()
        self.dim = dim
        self.freq_embed = nn.Parameter(torch.empty(1, max_freq_patches, dim))
        self.time_embed = nn.Parameter(torch.empty(1, max_time_patches, dim))
        _trunc_normal_(self.freq_embed, std=0.02)
        _trunc_normal_(self.time_embed, std=0.02)

    def forward(self, x: torch.Tensor, height: int, width: int) -> torch.Tensor:
        if height > self.freq_embed.size(1) or width > self.time_embed.size(1):
            raise ValueError(
                f"grid ({height}, {width}) exceeds max "
                f"({self.freq_embed.size(1)}, {self.time_embed.size(1)})"
            )
        fe = self.freq_embed[:, :height, :].unsqueeze(2)
        te = self.time_embed[:, :width, :].unsqueeze(1)
        pe = fe + te
        pe = pe.reshape(1, height * width, self.dim)
        return x + pe


class CLSToken(nn.Module):
    """Prepend a class token.

    Shapes:
        Input: (batch, seq, dim)
        Output: (batch, seq + 1, dim)
    """

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.cls_token = nn.Parameter(torch.empty(1, 1, dim))
        _trunc_normal_(self.cls_token, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b = x.size(0)
        cls = self.cls_token.expand(b, -1, -1)
        return torch.cat([cls, x], dim=1)
