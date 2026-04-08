from __future__ import annotations

import pytest
import torch

from src.data.augmentation import FrequencyMask, GainJitter, Mixup, TimeMask, TimeShift


def test_time_shift_preserves_shape() -> None:
    torch.manual_seed(0)
    ts = TimeShift(max_shift_samples=32, p=1.0)
    x = torch.randn(2, 1, 4000)
    y = ts(x)
    assert y.shape == x.shape


def test_time_shift_zero_max() -> None:
    torch.manual_seed(0)
    ts = TimeShift(max_shift_samples=0, p=1.0)
    x = torch.randn(2, 1, 500)
    y = ts(x)
    assert torch.equal(y, x)


def test_time_shift_probability_zero() -> None:
    torch.manual_seed(0)
    ts = TimeShift(max_shift_samples=100, p=0.0)
    x = torch.randn(2, 1, 800)
    y = ts(x)
    assert torch.equal(y, x)


def test_gain_jitter_preserves_shape() -> None:
    torch.manual_seed(0)
    gj = GainJitter(min_db=-6.0, max_db=6.0, p=1.0)
    x = torch.randn(3, 2, 1000)
    y = gj(x)
    assert y.shape == x.shape


def test_gain_jitter_probability_zero() -> None:
    torch.manual_seed(0)
    gj = GainJitter(p=0.0)
    x = torch.randn(2, 100)
    y = gj(x)
    assert torch.equal(y, x)


def test_gain_jitter_changes_magnitude() -> None:
    changed = False
    for seed in range(64):
        torch.manual_seed(seed)
        gj = GainJitter(min_db=-6.0, max_db=6.0, p=1.0)
        x = torch.ones(2, 512)
        y = gj(x)
        if not torch.allclose(x, y):
            changed = True
            break
    assert changed


def test_freq_mask_preserves_shape() -> None:
    torch.manual_seed(0)
    fm = FrequencyMask(max_width=20, p=1.0)
    x = torch.randn(2, 1, 128, 64)
    y = fm(x)
    assert y.shape == x.shape


def test_freq_mask_zeros_some_bins() -> None:
    torch.manual_seed(123)
    fm = FrequencyMask(max_width=30, p=1.0)
    x = torch.randn(2, 1, 128, 40) + 1.0
    y = fm(x)
    per_mel_zeroed = (y.abs().sum(dim=-1) == 0).squeeze(1)
    assert per_mel_zeroed.any(dim=-1).all()


def test_freq_mask_zero_width() -> None:
    torch.manual_seed(0)
    fm = FrequencyMask(max_width=0, p=1.0)
    x = torch.randn(2, 1, 128, 50)
    y = fm(x)
    assert torch.equal(y, x)


def test_time_mask_preserves_shape() -> None:
    torch.manual_seed(0)
    tm = TimeMask(max_width=10, p=1.0)
    x = torch.randn(4, 1, 64, 100)
    y = tm(x)
    assert y.shape == x.shape


def test_time_mask_zeros_some_frames() -> None:
    torch.manual_seed(456)
    tm = TimeMask(max_width=15, p=1.0)
    x = torch.randn(2, 1, 64, 80) + 1.0
    y = tm(x)
    per_frame_zeroed = (y.abs().sum(dim=-2) == 0).squeeze(1)
    assert per_frame_zeroed.any(dim=-1).all()


def test_time_mask_zero_width() -> None:
    torch.manual_seed(0)
    tm = TimeMask(max_width=0, p=1.0)
    x = torch.randn(2, 1, 64, 60)
    y = tm(x)
    assert torch.equal(y, x)


def test_mixup_output_shape() -> None:
    torch.manual_seed(0)
    mix = Mixup(alpha=0.4)
    a = torch.randn(5, 1, 32, 48)
    b = torch.randn(5, 1, 32, 48)
    mixed, weights = mix(a, b)
    assert mixed.shape == a.shape
    assert weights.shape == (5, 2)


def test_mixup_weights_sum_to_one() -> None:
    torch.manual_seed(0)
    mix = Mixup(alpha=1.0)
    a = torch.randn(4, 1, 16, 24)
    b = torch.randn(4, 1, 16, 24)
    _, weights = mix(a, b)
    sums = weights[:, 0] + weights[:, 1]
    for s in sums.tolist():
        assert s == pytest.approx(1.0, rel=0.0, abs=1e-5)


def test_mixup_shape_mismatch_raises() -> None:
    torch.manual_seed(0)
    mix = Mixup(alpha=0.5)
    a = torch.randn(2, 1, 32, 10)
    b = torch.randn(2, 1, 32, 11)
    with pytest.raises(ValueError, match="same shape"):
        mix(a, b)


def test_mixup_result_between_inputs() -> None:
    torch.manual_seed(99)
    mix = Mixup(alpha=0.3)
    a = torch.randn(3, 1, 24, 16)
    b = torch.randn(3, 1, 24, 16)
    mixed, _ = mix(a, b)
    lo = torch.minimum(a, b)
    hi = torch.maximum(a, b)
    assert (mixed >= lo - 1e-5).all()
    assert (mixed <= hi + 1e-5).all()


def test_freq_mask_probability_zero() -> None:
    torch.manual_seed(0)
    fm = FrequencyMask(max_width=40, p=0.0)
    x = torch.randn(2, 1, 128, 70)
    y = fm(x)
    assert torch.equal(y, x)


def test_time_mask_probability_zero() -> None:
    torch.manual_seed(0)
    tm = TimeMask(max_width=25, p=0.0)
    x = torch.randn(2, 1, 64, 90)
    y = tm(x)
    assert torch.equal(y, x)
