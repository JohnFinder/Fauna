from __future__ import annotations

import torch

from src.model.patch_embed import CLSToken, ConvFrontend, PatchEmbedding, PositionalEncoding2D


def test_conv_frontend_output_shape() -> None:
    torch.manual_seed(42)
    dim = 32
    m = ConvFrontend(dim)
    x = torch.randn(2, 1, 128, 64)
    y = m(x)
    assert y.shape == (2, dim, 128, 64)


def test_conv_frontend_preserves_spatial() -> None:
    torch.manual_seed(42)
    m = ConvFrontend(48)
    x = torch.randn(2, 1, 128, 64)
    y = m(x)
    assert y.shape[2] == x.shape[2] and y.shape[3] == x.shape[3]


def test_patch_embed_output_shape() -> None:
    torch.manual_seed(42)
    n_mels, pf, pt, dim, dim_in = 128, 16, 4, 32, 32
    m = PatchEmbedding(
        n_mels=n_mels,
        patch_freq=pf,
        patch_time=pt,
        dim=dim,
        dim_in=dim_in,
    )
    t = 64
    x = torch.randn(2, dim_in, n_mels, t)
    y = m(x)
    num_patches = (n_mels // pf) * (t // pt)
    assert y.shape == (2, num_patches, dim)


def test_patch_embed_rejects_wrong_mels() -> None:
    torch.manual_seed(42)
    try:
        PatchEmbedding(n_mels=127, patch_freq=16, patch_time=4, dim=32, dim_in=32)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError")


def test_patch_embed_rejects_wrong_time() -> None:
    torch.manual_seed(42)
    m = PatchEmbedding(
        n_mels=128,
        patch_freq=16,
        patch_time=4,
        dim=32,
        dim_in=32,
    )
    x = torch.randn(2, 32, 128, 63)
    try:
        m(x)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError")


def test_patch_embed_rejects_wrong_channels() -> None:
    torch.manual_seed(42)
    m = PatchEmbedding(
        n_mels=128,
        patch_freq=16,
        patch_time=4,
        dim=32,
        dim_in=32,
    )
    x = torch.randn(2, 31, 128, 64)
    try:
        m(x)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError")


def test_pos_encoding_adds_to_input() -> None:
    torch.manual_seed(42)
    dim = 32
    pe = PositionalEncoding2D(dim, max_freq_patches=8, max_time_patches=32)
    x = torch.randn(2, 24, dim)
    y = pe(x, height=4, width=6)
    assert not torch.allclose(y, x)


def test_pos_encoding_rejects_too_large_grid() -> None:
    torch.manual_seed(42)
    pe = PositionalEncoding2D(32, max_freq_patches=4, max_time_patches=4)
    x = torch.randn(1, 25, 32)
    try:
        pe(x, height=5, width=5)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError")


def test_pos_encoding_shape_preserved() -> None:
    torch.manual_seed(42)
    pe = PositionalEncoding2D(48, max_freq_patches=8, max_time_patches=16)
    x = torch.randn(3, 40, 48)
    y = pe(x, height=5, width=8)
    assert y.shape == x.shape


def test_cls_token_adds_one() -> None:
    torch.manual_seed(42)
    cls = CLSToken(32)
    x = torch.randn(2, 10, 32)
    y = cls(x)
    assert y.shape == (2, 11, 32)


def test_cls_token_first_position() -> None:
    torch.manual_seed(42)
    dim = 32
    cls_mod = CLSToken(dim)
    x = torch.randn(4, 7, dim)
    y = cls_mod(x)
    expected = cls_mod.cls_token.expand(4, -1, -1)
    assert torch.allclose(y[:, 0, :], expected.squeeze(1))


def test_cls_token_batch_independent() -> None:
    torch.manual_seed(42)
    dim = 48
    cls_mod = CLSToken(dim)
    x = torch.randn(3, 5, dim)
    y = cls_mod(x)
    assert torch.allclose(y[0, 0], y[1, 0])
    assert torch.allclose(y[1, 0], y[2, 0])
