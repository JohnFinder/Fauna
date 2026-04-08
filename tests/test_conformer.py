from __future__ import annotations

import torch

from src.model.conformer import (
    ConformerBlock,
    ConvolutionModule,
    FeedForwardModule,
    MultiHeadSelfAttention,
    Swish,
)


def test_swish_zero_is_zero() -> None:
    torch.manual_seed(42)
    m = Swish()
    x = torch.zeros(3, 4)
    assert torch.allclose(m(x), torch.zeros_like(x))


def test_swish_positive() -> None:
    torch.manual_seed(42)
    m = Swish()
    x = torch.tensor([0.1, 1.0, 2.0])
    assert bool(torch.all(m(x) > 0))


def test_swish_gradient_exists() -> None:
    torch.manual_seed(42)
    m = Swish()
    x = torch.randn(2, 3, requires_grad=True)
    y = m(x).sum()
    y.backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()


def test_ffn_output_shape() -> None:
    torch.manual_seed(42)
    dim, ffn_dim, b, t = 32, 64, 2, 17
    m = FeedForwardModule(dim, ffn_dim, dropout=0.0)
    x = torch.randn(b, t, dim)
    y = m(x)
    assert y.shape == (b, t, dim)


def test_ffn_zero_dropout() -> None:
    torch.manual_seed(42)
    dim, ffn_dim = 32, 64
    m = FeedForwardModule(dim, ffn_dim, dropout=0.0)
    m.eval()
    x = torch.randn(2, 5, dim)
    y1 = m(x)
    y2 = m(x)
    assert torch.equal(y1, y2)


def test_mhsa_output_shape() -> None:
    torch.manual_seed(42)
    dim, heads, b, t = 32, 2, 2, 11
    m = MultiHeadSelfAttention(dim, heads, dropout=0.0)
    x = torch.randn(b, t, dim)
    y = m(x)
    assert y.shape == (b, t, dim)


def test_mhsa_rejects_indivisible_dim() -> None:
    torch.manual_seed(42)
    try:
        MultiHeadSelfAttention(dim=33, num_heads=4, dropout=0.0)
    except ValueError as e:
        assert "33" in str(e) and "4" in str(e)
    else:
        raise AssertionError("expected ValueError")


def test_mhsa_different_seq_lengths() -> None:
    torch.manual_seed(42)
    dim, heads = 32, 2
    m = MultiHeadSelfAttention(dim, heads, dropout=0.0)
    for t in (1, 10, 50):
        x = torch.randn(2, t, dim)
        y = m(x)
        assert y.shape == (2, t, dim)


def test_conv_module_output_shape() -> None:
    torch.manual_seed(42)
    dim, k, b, t = 32, 3, 2, 9
    m = ConvolutionModule(dim, k, dropout=0.0)
    x = torch.randn(b, t, dim)
    y = m(x)
    assert y.shape == (b, t, dim)


def test_conv_module_rejects_even_kernel() -> None:
    torch.manual_seed(42)
    try:
        ConvolutionModule(dim=32, kernel_size=4, dropout=0.0)
    except ValueError as e:
        assert "4" in str(e) or "odd" in str(e).lower()
    else:
        raise AssertionError("expected ValueError")


def test_conv_module_different_kernels() -> None:
    torch.manual_seed(42)
    dim, b, t = 32, 2, 7
    for k in (3, 7, 15):
        m = ConvolutionModule(dim, k, dropout=0.0)
        x = torch.randn(b, t, dim)
        y = m(x)
        assert y.shape == (b, t, dim)


def test_conformer_block_output_shape() -> None:
    torch.manual_seed(42)
    blk = ConformerBlock(
        dim=32,
        num_heads=2,
        ffn_dim=64,
        conv_kernel_size=3,
        dropout=0.0,
    )
    x = torch.randn(2, 11, 32)
    y = blk(x)
    assert y.shape == x.shape


def test_conformer_block_residual_connection() -> None:
    torch.manual_seed(42)
    blk = ConformerBlock(
        dim=32,
        num_heads=2,
        ffn_dim=64,
        conv_kernel_size=3,
        dropout=0.0,
    )

    def zero_linear_conv(module: torch.nn.Module) -> None:
        for child in module.modules():
            if isinstance(child, torch.nn.Linear):
                torch.nn.init.zeros_(child.weight)
                if child.bias is not None:
                    torch.nn.init.zeros_(child.bias)
            if isinstance(child, torch.nn.Conv1d):
                torch.nn.init.zeros_(child.weight)
                if child.bias is not None:
                    torch.nn.init.zeros_(child.bias)

    for sub in (blk.ffn1, blk.mhsa, blk.conv, blk.ffn2):
        zero_linear_conv(sub)

    x = torch.randn(2, 8, 32)
    y = blk(x)
    assert not torch.allclose(y, torch.zeros_like(y))


def test_conformer_block_is_differentiable() -> None:
    torch.manual_seed(42)
    blk = ConformerBlock(
        dim=32,
        num_heads=2,
        ffn_dim=64,
        conv_kernel_size=3,
        dropout=0.0,
    )
    x = torch.randn(2, 6, 32, requires_grad=True)
    y = blk(x).sum()
    y.backward()
    assert x.grad is not None


def test_conformer_block_eval_mode() -> None:
    torch.manual_seed(42)
    blk = ConformerBlock(
        dim=32,
        num_heads=2,
        ffn_dim=64,
        conv_kernel_size=3,
        dropout=0.0,
    )
    blk.eval()
    x = torch.randn(3, 10, 32)
    y = blk(x)
    assert y.shape == x.shape
    assert torch.isfinite(y).all()
