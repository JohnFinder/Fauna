from __future__ import annotations

import torch

from src.model.budgiformer import (
    BUDGIFORMER_M,
    BUDGIFORMER_S,
    BudgiFormer,
    BudgiFormerConfig,
)


def test_budgiformer_s_config_defaults() -> None:
    torch.manual_seed(42)
    c = BudgiFormerConfig()
    assert c.n_mels == 128
    assert c.patch_freq == 16
    assert c.patch_time == 4
    assert c.dim == 256
    assert c.num_layers == 8
    assert c.num_heads == 4
    assert c.ffn_dim == 1024
    assert c.conv_kernel_size == 31
    assert c.dropout == 0.1
    assert c.max_freq_patches == 8
    assert c.max_time_patches == 64


def test_budgiformer_m_config_values() -> None:
    torch.manual_seed(42)
    m = BUDGIFORMER_M
    assert m.dim == 384
    assert m.num_layers == 12
    assert m.num_heads == 6
    assert m.ffn_dim == 1536
    assert m.n_mels == BUDGIFORMER_S.n_mels
    assert m.patch_freq == BUDGIFORMER_S.patch_freq
    assert m.patch_time == BUDGIFORMER_S.patch_time


def _small_config() -> BudgiFormerConfig:
    return BudgiFormerConfig(
        n_mels=128,
        patch_freq=16,
        patch_time=4,
        dim=32,
        num_layers=1,
        num_heads=2,
        ffn_dim=64,
        conv_kernel_size=3,
        dropout=0.0,
        max_freq_patches=8,
        max_time_patches=128,
    )


def test_forward_output_shape() -> None:
    torch.manual_seed(42)
    cfg = _small_config()
    model = BudgiFormer(cfg)
    model.eval()
    b, time = 2, 64
    spec = torch.randn(b, 1, 128, time)
    y = model(spec)
    nf = cfg.n_mels // cfg.patch_freq
    nt = time // cfg.patch_time
    num_patches = nf * nt
    assert y.shape == (b, num_patches + 1, cfg.dim)


def test_cls_embedding_shape() -> None:
    torch.manual_seed(42)
    cfg = _small_config()
    model = BudgiFormer(cfg)
    model.eval()
    spec = torch.randn(2, 1, 128, 64)
    cls_e = model.get_cls_embedding(spec)
    assert cls_e.shape == (2, cfg.dim)


def test_patch_embeddings_shape() -> None:
    torch.manual_seed(42)
    cfg = _small_config()
    model = BudgiFormer(cfg)
    model.eval()
    spec = torch.randn(2, 1, 128, 64)
    pe = model.get_patch_embeddings(spec)
    nf = cfg.n_mels // cfg.patch_freq
    nt = 64 // cfg.patch_time
    assert pe.shape == (2, nf * nt, cfg.dim)


def test_cls_plus_patches_equals_forward() -> None:
    torch.manual_seed(42)
    cfg = _small_config()
    model = BudgiFormer(cfg)
    model.eval()
    spec = torch.randn(2, 1, 128, 64)
    full = model(spec)
    cls_e = model.get_cls_embedding(spec)
    patches = model.get_patch_embeddings(spec)
    assert torch.allclose(full[:, 0, :], cls_e)
    assert torch.allclose(full[:, 1:, :], patches)


def test_different_time_lengths() -> None:
    torch.manual_seed(42)
    cfg = _small_config()
    model = BudgiFormer(cfg)
    model.eval()
    for time in (32, 64, 128):
        spec = torch.randn(2, 1, 128, time)
        y = model(spec)
        nf = cfg.n_mels // cfg.patch_freq
        nt = time // cfg.patch_time
        assert y.shape == (2, nf * nt + 1, cfg.dim)


def test_single_batch() -> None:
    torch.manual_seed(42)
    cfg = _small_config()
    model = BudgiFormer(cfg)
    model.eval()
    spec = torch.randn(1, 1, 128, 64)
    y = model(spec)
    assert y.shape[0] == 1


def test_gradient_flows() -> None:
    torch.manual_seed(42)
    cfg = _small_config()
    model = BudgiFormer(cfg)
    model.train()
    spec = torch.randn(2, 1, 128, 64)
    cls_e = model.get_cls_embedding(spec)
    loss = cls_e.pow(2).mean()
    loss.backward()
    assert loss.ndim == 0


def test_parameter_count_reasonable() -> None:
    torch.manual_seed(42)
    model = BudgiFormer(BUDGIFORMER_S)
    n = sum(p.numel() for p in model.parameters())
    assert n > 1_000_000
    assert n < 50_000_000
