from __future__ import annotations

import pytest
import torch

from src.data.frontend import MelSpectrogramFrontend, _preemphasis, pad_or_trim


def _log_mel_before_norm(fe: MelSpectrogramFrontend, waveform: torch.Tensor) -> torch.Tensor:
    if waveform.dim() == 2:
        waveform = waveform.unsqueeze(1)
    x = _preemphasis(waveform, fe.preemph_coeff)
    mel = fe.mel(x.squeeze(1))
    return torch.log(mel.clamp_min(fe.log_eps)).unsqueeze(1)


def test_pad_or_trim_pads_short() -> None:
    torch.manual_seed(0)
    w = torch.randn(2, 50)
    out = pad_or_trim(w, 100, dim=-1)
    assert out.shape == (2, 100)
    assert torch.allclose(out[:, :50], w)
    assert torch.all(out[:, 50:] == 0)


def test_pad_or_trim_trims_long() -> None:
    torch.manual_seed(0)
    w = torch.randn(2, 200)
    out = pad_or_trim(w, 80, dim=-1)
    assert out.shape == (2, 80)
    assert torch.allclose(out, w[:, :80])


def test_pad_or_trim_exact_length_unchanged() -> None:
    torch.manual_seed(0)
    w = torch.randn(3, 64)
    out = pad_or_trim(w, 64, dim=-1)
    assert out.shape == w.shape
    assert out is w or torch.equal(out, w)


def test_pad_or_trim_custom_dim() -> None:
    torch.manual_seed(0)
    w = torch.randn(2, 5, 4)
    out = pad_or_trim(w, 10, dim=1)
    assert out.shape == (2, 10, 4)
    assert torch.allclose(out[:, :5, :], w)
    assert torch.all(out[:, 5:, :] == 0)


def test_preemphasis_first_sample_unchanged() -> None:
    torch.manual_seed(0)
    w = torch.randn(2, 8)
    out = _preemphasis(w, 0.97)
    assert torch.allclose(out[..., :1], w[..., :1])


def test_preemphasis_subsequent_samples() -> None:
    coeff = 0.97
    w = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    out = _preemphasis(w, coeff)
    expected = torch.cat(
        (w[..., :1], w[..., 1:] - coeff * w[..., :-1]),
        dim=-1,
    )
    assert torch.allclose(out, expected)


def test_preemphasis_single_sample() -> None:
    w = torch.randn(2, 1)
    out = _preemphasis(w, 0.97)
    assert torch.allclose(out, w)


def test_frontend_output_shape_2d_input() -> None:
    torch.manual_seed(0)
    fe = MelSpectrogramFrontend(
        sample_rate=48000,
        n_fft=240,
        hop_length=120,
        n_mels=128,
        f_min=500.0,
        f_max=10000.0,
        preemph_coeff=0.97,
    )
    wave = torch.randn(4, 4800)
    ref = _log_mel_before_norm(fe, wave)
    out = fe(wave)
    assert out.shape == (4, 1, 128, ref.shape[-1])


def test_frontend_output_shape_3d_input() -> None:
    torch.manual_seed(0)
    fe = MelSpectrogramFrontend(
        sample_rate=48000,
        n_fft=240,
        hop_length=120,
        n_mels=128,
        f_min=500.0,
        f_max=10000.0,
        preemph_coeff=0.97,
    )
    wave = torch.randn(3, 1, 4800)
    ref = _log_mel_before_norm(fe, wave)
    out = fe(wave)
    assert out.shape == (3, 1, 128, ref.shape[-1])


def test_frontend_rejects_wrong_channels() -> None:
    fe = MelSpectrogramFrontend()
    bad = torch.randn(2, 2, 1000)
    with pytest.raises(ValueError, match="waveform must be"):
        fe(bad)


def test_frontend_rejects_4d() -> None:
    fe = MelSpectrogramFrontend()
    bad = torch.randn(2, 1, 1, 1000)
    with pytest.raises(ValueError, match="waveform must be"):
        fe(bad)


def test_frontend_output_is_finite() -> None:
    torch.manual_seed(0)
    fe = MelSpectrogramFrontend()
    wave = torch.randn(2, 1, 5000)
    out = fe(wave)
    assert torch.isfinite(out).all()


def test_frontend_fixed_normalization() -> None:
    torch.manual_seed(0)
    n_mels = 128
    mean = torch.linspace(-1.0, 1.0, n_mels)
    std = torch.linspace(0.5, 1.5, n_mels)
    fe = MelSpectrogramFrontend(
        n_mels=n_mels,
        channel_mean=mean,
        channel_std=std,
    )
    wave = torch.randn(2, 4000)
    log_mel = _log_mel_before_norm(fe, wave)
    expected = (log_mel - fe.fixed_mean) / fe.fixed_std
    out = fe(wave)
    assert torch.allclose(out, expected)


def test_frontend_running_stats_training() -> None:
    torch.manual_seed(42)
    fe = MelSpectrogramFrontend(track_running_stats=True)
    fe.train()
    assert int(fe.num_batches_tracked.item()) == 0
    w = torch.randn(2, 1, 6000)
    log_mel = _log_mel_before_norm(fe, w)
    batch_mean = log_mel.mean(dim=(0, -1), keepdim=True).view(1, fe.n_mels, 1)
    _ = fe(w)
    assert int(fe.num_batches_tracked.item()) == 1
    assert torch.allclose(fe.running_mean, batch_mean)
    w2 = torch.randn(2, 1, 6000)
    log_mel2 = _log_mel_before_norm(fe, w2)
    mean2 = log_mel2.mean(dim=(0, -1), keepdim=True).view(1, fe.n_mels, 1)
    rm_before = fe.running_mean.clone()
    _ = fe(w2)
    expected_running = rm_before * (1 - fe.momentum) + mean2 * fe.momentum
    assert torch.allclose(fe.running_mean, expected_running)


def test_frontend_running_stats_eval() -> None:
    torch.manual_seed(7)
    fe = MelSpectrogramFrontend(track_running_stats=True)
    fe.train()
    for _ in range(3):
        fe(torch.randn(2, 1, 5000))
    fe.eval()
    w = torch.randn(2, 1, 5000)
    log_mel = _log_mel_before_norm(fe, w)
    std = (fe.running_var + fe.norm_eps).sqrt()
    expected = (log_mel - fe.running_mean) / std
    out = fe(w)
    assert torch.allclose(out, expected)


def test_frontend_default_normalization() -> None:
    torch.manual_seed(0)
    fe = MelSpectrogramFrontend(track_running_stats=False)
    wave = torch.randn(3, 1, 8000)
    out = fe(wave)
    log_mel = _log_mel_before_norm(fe, wave)
    mean = log_mel.mean(dim=-1, keepdim=True)
    std = log_mel.std(dim=-1, keepdim=True).clamp_min(fe.norm_eps)
    expected = (log_mel - mean) / std
    assert torch.allclose(out, expected)


def test_frontend_different_batch_sizes() -> None:
    torch.manual_seed(0)
    fe = MelSpectrogramFrontend()
    for b in (1, 8):
        wave = torch.randn(b, 3200)
        ref = _log_mel_before_norm(fe, wave)
        out = fe(wave)
        assert out.shape == (b, 1, 128, ref.shape[-1])
        assert torch.isfinite(out).all()
