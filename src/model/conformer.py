from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class Swish(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(x)


def _relative_sinusoidal(seq_len: int, dim: int, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    num_pos = 2 * seq_len - 1
    position = torch.arange(-(seq_len - 1), seq_len, device=device, dtype=dtype).unsqueeze(1)
    div_term = torch.exp(
        torch.arange(0, dim, 2, device=device, dtype=dtype) * (-math.log(10000.0) / dim)
    )
    pe = torch.zeros(num_pos, dim, device=device, dtype=dtype)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe


class FeedForwardModule(nn.Module):
    def __init__(self, dim: int, ffn_dim: int, dropout: float) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.linear1 = nn.Linear(dim, ffn_dim)
        self.act = Swish()
        self.dropout1 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(ffn_dim, dim)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(x)
        x = self.linear1(x)
        x = self.act(x)
        x = self.dropout1(x)
        x = self.linear2(x)
        x = self.dropout2(x)
        return x


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim ({dim}) must be divisible by num_heads ({num_heads})")
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5

        self.norm = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, 3 * dim, bias=True)
        self.rel_pos_proj = nn.Linear(self.head_dim, num_heads, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, _ = x.shape
        x = self.norm(x)
        qkv = self.qkv(x).reshape(b, t, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        scores = (q @ k.transpose(-2, -1)) * self.scale

        rel_pe = _relative_sinusoidal(t, self.head_dim, scores.dtype, scores.device)
        rel_bias = self.rel_pos_proj(rel_pe)
        idx = torch.arange(t, device=x.device).unsqueeze(1) - torch.arange(t, device=x.device).unsqueeze(0)
        idx = idx + (t - 1)
        bias = rel_bias[idx].permute(2, 0, 1).unsqueeze(0)
        scores = scores + bias

        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        out = attn @ v
        out = out.transpose(1, 2).reshape(b, t, self.dim)
        out = self.out_proj(out)
        return out


class ConvolutionModule(nn.Module):
    def __init__(self, dim: int, kernel_size: int, dropout: float) -> None:
        super().__init__()
        if kernel_size % 2 != 1:
            raise ValueError(f"kernel_size ({kernel_size}) must be odd for same-length padding")
        self.norm = nn.LayerNorm(dim)
        self.pw1 = nn.Conv1d(dim, 2 * dim, kernel_size=1)
        self.dw = nn.Conv1d(
            dim,
            dim,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=dim,
        )
        self.bn = nn.BatchNorm1d(dim)
        self.act = Swish()
        self.pw2 = nn.Conv1d(dim, dim, kernel_size=1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(x)
        x = x.transpose(1, 2)
        x = self.pw1(x)
        a, b = x.chunk(2, dim=1)
        x = a * torch.sigmoid(b)
        x = self.dw(x)
        x = self.bn(x)
        x = self.act(x)
        x = self.pw2(x)
        x = x.transpose(1, 2)
        x = self.dropout(x)
        return x


class ConformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        ffn_dim: int,
        conv_kernel_size: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.norm_ffn1 = nn.LayerNorm(dim)
        self.ffn1 = FeedForwardModule(dim, ffn_dim, dropout)
        self.norm_mhsa = nn.LayerNorm(dim)
        self.mhsa = MultiHeadSelfAttention(dim, num_heads, dropout)
        self.norm_conv = nn.LayerNorm(dim)
        self.conv = ConvolutionModule(dim, conv_kernel_size, dropout)
        self.norm_ffn2 = nn.LayerNorm(dim)
        self.ffn2 = FeedForwardModule(dim, ffn_dim, dropout)
        self.norm_out = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + 0.5 * self.ffn1(self.norm_ffn1(x))
        x = x + self.mhsa(self.norm_mhsa(x))
        x = x + self.conv(self.norm_conv(x))
        x = self.norm_out(x + 0.5 * self.ffn2(self.norm_ffn2(x)))
        return x
