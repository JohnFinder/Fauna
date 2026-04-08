from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from .conformer import ConformerBlock
from .patch_embed import CLSToken, ConvFrontend, PatchEmbedding, PositionalEncoding2D


@dataclass
class BudgiFormerConfig:
    n_mels: int = 128
    patch_freq: int = 16
    patch_time: int = 4
    dim: int = 256
    num_layers: int = 8
    num_heads: int = 4
    ffn_dim: int = 1024
    conv_kernel_size: int = 31
    dropout: float = 0.1
    max_freq_patches: int = 8
    max_time_patches: int = 64


BUDGIFORMER_S = BudgiFormerConfig()
BUDGIFORMER_M = BudgiFormerConfig(dim=384, num_layers=12, num_heads=6, ffn_dim=1536)


class BudgiFormer(nn.Module):
    """Conformer encoder over patch tokens and a prepended CLS.

    Shapes:
        forward(spectrogram): input (batch, 1, n_mels, time), output (batch, num_patches + 1, dim)
        get_cls_embedding: output (batch, dim)
        get_patch_embeddings: output (batch, num_patches, dim)
    """

    def __init__(self, config: BudgiFormerConfig) -> None:
        super().__init__()
        self.config = config
        self.conv_frontend = ConvFrontend(config.dim)
        self.patch_embed = PatchEmbedding(
            n_mels=config.n_mels,
            patch_freq=config.patch_freq,
            patch_time=config.patch_time,
            dim=config.dim,
            dim_in=config.dim,
        )
        self.pos_encode = PositionalEncoding2D(
            config.dim,
            max_freq_patches=config.max_freq_patches,
            max_time_patches=config.max_time_patches,
        )
        self.cls_token = CLSToken(config.dim)
        self.blocks = nn.ModuleList(
            ConformerBlock(
                dim=config.dim,
                num_heads=config.num_heads,
                ffn_dim=config.ffn_dim,
                conv_kernel_size=config.conv_kernel_size,
                dropout=config.dropout,
            )
            for _ in range(config.num_layers)
        )
        self.norm = nn.LayerNorm(config.dim)

    def forward(self, spectrogram: torch.Tensor) -> torch.Tensor:
        x = self.conv_frontend(spectrogram)
        x = self.patch_embed(x)
        num_freq = self.patch_embed.num_freq_patches
        num_time = x.shape[1] // num_freq
        x = self.pos_encode(x, num_freq, num_time)
        x = self.cls_token(x)
        for block in self.blocks:
            x = block(x)
        return self.norm(x)

    def get_cls_embedding(self, spectrogram: torch.Tensor) -> torch.Tensor:
        return self.forward(spectrogram)[:, 0, :]

    def get_patch_embeddings(self, spectrogram: torch.Tensor) -> torch.Tensor:
        return self.forward(spectrogram)[:, 1:, :]
