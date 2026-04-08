from __future__ import annotations

import torch
import torch.nn as nn
import torchaudio


def pad_or_trim(waveform: torch.Tensor, target_length: int, dim: int = -1) -> torch.Tensor:
    if dim < 0:
        dim = waveform.ndim + dim
    length = waveform.shape[dim]
    if length == target_length:
        return waveform
    if length > target_length:
        slc = [slice(None)] * waveform.ndim
        slc[dim] = slice(0, target_length)
        return waveform[tuple(slc)]
    pad_len = target_length - length
    pad_pattern = [0] * (2 * waveform.ndim)
    pad_idx = 2 * (waveform.ndim - 1 - dim)
    pad_pattern[pad_idx + 1] = pad_len
    return torch.nn.functional.pad(waveform, pad_pattern, mode="constant", value=0.0)


def _preemphasis(waveform: torch.Tensor, coeff: float) -> torch.Tensor:
    if waveform.shape[-1] < 2:
        return waveform
    return torch.cat(
        (waveform[..., :1], waveform[..., 1:] - coeff * waveform[..., :-1]),
        dim=-1,
    )


class MelSpectrogramFrontend(nn.Module):
    def __init__(
        self,
        sample_rate: int = 48000,
        n_fft: int = 240,
        hop_length: int = 120,
        n_mels: int = 128,
        f_min: float = 500.0,
        f_max: float = 10000.0,
        preemph_coeff: float = 0.97,
        log_eps: float = 1e-6,
        norm_eps: float = 1e-5,
        track_running_stats: bool = False,
        momentum: float = 0.1,
        channel_mean: torch.Tensor | None = None,
        channel_std: torch.Tensor | None = None,
        center: bool = False,
    ) -> None:
        super().__init__()
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.f_min = f_min
        self.f_max = f_max
        self.preemph_coeff = preemph_coeff
        self.log_eps = log_eps
        self.norm_eps = norm_eps
        self.track_running_stats = track_running_stats
        self.momentum = momentum
        self.center = center

        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            f_min=f_min,
            f_max=f_max,
            center=center,
            power=2.0,
        )

        if channel_mean is not None and channel_std is not None:
            self.register_buffer("fixed_mean", channel_mean.reshape(1, n_mels, 1).float())
            self.register_buffer("fixed_std", channel_std.reshape(1, n_mels, 1).float().clamp_min(norm_eps))
        else:
            self.register_buffer("fixed_mean", torch.empty(0))
            self.register_buffer("fixed_std", torch.empty(0))

        if track_running_stats:
            self.register_buffer("running_mean", torch.zeros(1, n_mels, 1))
            self.register_buffer("running_var", torch.ones(1, n_mels, 1))
            self.register_buffer("num_batches_tracked", torch.tensor(0, dtype=torch.long))
        else:
            self.register_buffer("running_mean", torch.empty(0))
            self.register_buffer("running_var", torch.empty(0))
            self.register_buffer("num_batches_tracked", torch.tensor(0, dtype=torch.long))

    def _normalize(
        self,
        log_mel: torch.Tensor,
    ) -> torch.Tensor:
        if self.fixed_mean.numel() > 0 and self.fixed_std.numel() > 0:
            return (log_mel - self.fixed_mean) / self.fixed_std

        if self.track_running_stats and self.running_mean.numel() > 0:
            if self.training:
                batch_mean = log_mel.mean(dim=(0, -1), keepdim=True)
                batch_var = log_mel.var(dim=(0, -1), unbiased=False, keepdim=True)
                new_mean = batch_mean.view(1, self.n_mels, 1)
                new_var = batch_var.view(1, self.n_mels, 1)
                with torch.no_grad():
                    if int(self.num_batches_tracked.item()) == 0:
                        self.running_mean.copy_(new_mean)
                        self.running_var.copy_(new_var)
                    else:
                        self.running_mean.mul_(1 - self.momentum).add_(new_mean * self.momentum)
                        self.running_var.mul_(1 - self.momentum).add_(new_var * self.momentum)
                    self.num_batches_tracked += 1
                std = (batch_var + self.norm_eps).sqrt()
                return (log_mel - batch_mean) / std
            std = (self.running_var + self.norm_eps).sqrt()
            return (log_mel - self.running_mean) / std

        mean = log_mel.mean(dim=-1, keepdim=True)
        std = log_mel.std(dim=-1, keepdim=True).clamp_min(self.norm_eps)
        return (log_mel - mean) / std

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        if waveform.dim() == 2:
            waveform = waveform.unsqueeze(1)
        if waveform.dim() != 3 or waveform.shape[1] != 1:
            raise ValueError("waveform must be (batch, time) or (batch, 1, time)")

        x = _preemphasis(waveform, self.preemph_coeff)
        mel = self.mel(x.squeeze(1))
        log_mel = torch.log(mel.clamp_min(self.log_eps))
        log_mel = log_mel.unsqueeze(1)
        log_mel = self._normalize(log_mel)
        return log_mel
