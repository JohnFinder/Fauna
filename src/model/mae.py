from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
from einops import rearrange

from .budgiformer import BudgiFormer, BudgiFormerConfig

__all__ = [
    "TransformerDecoderBlock",
    "MAEDecoder",
    "BudgiFormerMAE",
    "BudgiFormer",
    "BudgiFormerConfig",
]


class TransformerDecoderBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            dim,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.norm1(x)
        y, _ = self.attn(y, y, y, need_weights=False)
        x = x + y
        y = self.norm2(x)
        x = x + self.mlp(y)
        return x


class MAEDecoder(nn.Module):
    def __init__(
        self,
        encoder_dim: int,
        decoder_dim: int,
        num_layers: int = 4,
        num_heads: int = 4,
        patch_freq: int = 16,
        patch_time: int = 4,
        max_num_patches: int = 2048,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.decoder_dim = decoder_dim
        self.patch_freq = patch_freq
        self.patch_time = patch_time
        self.pred_dim = patch_freq * patch_time
        self.embed_dim_proj = nn.Linear(encoder_dim, decoder_dim)
        self.mask_token = nn.Parameter(torch.empty(1, 1, decoder_dim))
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        self.decoder_pos_embed = nn.Parameter(torch.empty(1, max_num_patches, decoder_dim))
        nn.init.trunc_normal_(self.decoder_pos_embed, std=0.02)
        self.decoder_blocks = nn.ModuleList(
            TransformerDecoderBlock(
                decoder_dim,
                num_heads,
                mlp_ratio=mlp_ratio,
                dropout=dropout,
            )
            for _ in range(num_layers)
        )
        self.decoder_norm = nn.LayerNorm(decoder_dim)
        self.decoder_pred = nn.Linear(decoder_dim, self.pred_dim)

    def forward(
        self,
        encoder_tokens: torch.Tensor,
        mask_indices: torch.Tensor,
        num_patches: int,
    ) -> torch.Tensor:
        b, len_keep, _ = encoder_tokens.shape
        if num_patches > self.decoder_pos_embed.size(1):
            raise ValueError(
                f"num_patches ({num_patches}) exceeds decoder_pos_embed length "
                f"({self.decoder_pos_embed.size(1)})",
            )
        num_mask = num_patches - len_keep
        z = self.embed_dim_proj(encoder_tokens)
        mask_tok = self.mask_token.expand(b, num_mask, -1)
        x_cat = torch.cat([z, mask_tok], dim=1)
        ids = mask_indices.unsqueeze(-1).expand(-1, -1, self.decoder_dim)
        x = torch.gather(x_cat, dim=1, index=ids)
        pe = self.decoder_pos_embed[:, :num_patches, :]
        x = x + pe
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)
        return self.decoder_pred(x)


class BudgiFormerMAE(nn.Module):
    def __init__(
        self,
        encoder: BudgiFormer,
        decoder: MAEDecoder,
        mask_ratio: float = 0.75,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.mask_ratio = mask_ratio

    def random_masking(
        self,
        sequence: torch.Tensor,
        mask_ratio: float,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        b, n, d = sequence.shape
        len_keep = max(1, int(n * (1.0 - mask_ratio)))
        noise = torch.rand(b, n, device=sequence.device, dtype=sequence.dtype)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        ids_keep = ids_shuffle[:, :len_keep]
        batch_idx = torch.arange(b, device=sequence.device).unsqueeze(1).expand_as(ids_keep)
        x_unmasked = sequence[batch_idx, ids_keep]
        mask = torch.ones(b, n, device=sequence.device)
        mask.scatter_(1, ids_keep, 0.0)
        return x_unmasked, mask, ids_restore.long()

    def _patch_targets(self, spectrogram: torch.Tensor) -> torch.Tensor:
        cfg = self.encoder.config
        return rearrange(
            spectrogram,
            "b c (hf pf) (wf pt) -> b (hf wf) (c pf pt)",
            pf=cfg.patch_freq,
            pt=cfg.patch_time,
        )

    def forward(
        self,
        spectrogram: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self.encoder.conv_frontend(spectrogram)
        x = self.encoder.patch_embed(x)
        num_patches = x.shape[1]
        x_unmasked, mask, ids_restore = self.random_masking(x, self.mask_ratio)
        for block in self.encoder.blocks:
            x_unmasked = block(x_unmasked)
        x_unmasked = self.encoder.norm(x_unmasked)
        recon = self.decoder(x_unmasked, ids_restore, num_patches)
        target = self._patch_targets(spectrogram)
        if target.shape[1] != num_patches:
            raise ValueError(f"target num_patches {target.shape[1]} != encoder {num_patches}")
        if target.shape[2] != self.decoder.pred_dim:
            raise ValueError(
                f"target patch dim {target.shape[2]} != decoder pred_dim {self.decoder.pred_dim}",
            )
        loss = (recon - target) ** 2
        loss = loss.mean(dim=-1)
        loss = (loss * mask).sum() / mask.clamp_min(1.0).sum()
        return loss, recon, mask
