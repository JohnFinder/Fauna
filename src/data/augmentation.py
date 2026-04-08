from __future__ import annotations

from typing import Tuple

import torch
import torch.nn.functional as F


class TimeShift:
    def __init__(self, max_shift_samples: int, p: float = 1.0) -> None:
        self.max_shift_samples = max_shift_samples
        self.p = p

    def __call__(self, waveform: torch.Tensor) -> torch.Tensor:
        if self.max_shift_samples <= 0 or torch.rand(1, device=waveform.device).item() > self.p:
            return waveform
        shift = int(torch.randint(-self.max_shift_samples, self.max_shift_samples + 1, (1,)).item())
        if shift == 0:
            return waveform
        if shift > 0:
            x = F.pad(waveform, (shift, 0), mode="constant", value=0.0)
            return x[..., :-shift]
        s = -shift
        x = F.pad(waveform, (0, s), mode="constant", value=0.0)
        return x[..., s:]


class GainJitter:
    def __init__(self, min_db: float = -6.0, max_db: float = 6.0, p: float = 1.0) -> None:
        self.min_db = min_db
        self.max_db = max_db
        self.p = p

    def __call__(self, waveform: torch.Tensor) -> torch.Tensor:
        if torch.rand(1, device=waveform.device).item() > self.p:
            return waveform
        u = torch.empty(1, device=waveform.device, dtype=waveform.dtype).uniform_(self.min_db, self.max_db)
        gain = torch.pow(torch.tensor(10.0, device=waveform.device, dtype=waveform.dtype), u / 20.0)
        return waveform * gain


class FrequencyMask:
    def __init__(self, max_width: int, p: float = 1.0) -> None:
        self.max_width = max_width
        self.p = p

    def __call__(self, spec: torch.Tensor) -> torch.Tensor:
        if self.max_width <= 0 or torch.rand(1, device=spec.device).item() > self.p:
            return spec
        _, _, n_mels, _ = spec.shape
        max_w = min(self.max_width, n_mels)
        if max_w < 1:
            return spec
        w = int(torch.randint(1, max_w + 1, (1,)).item())
        f0 = int(torch.randint(0, n_mels - w + 1, (1,)).item())
        out = spec.clone()
        out[:, :, f0 : f0 + w, :] = 0.0
        return out


class TimeMask:
    def __init__(self, max_width: int, p: float = 1.0) -> None:
        self.max_width = max_width
        self.p = p

    def __call__(self, spec: torch.Tensor) -> torch.Tensor:
        if self.max_width <= 0 or torch.rand(1, device=spec.device).item() > self.p:
            return spec
        t_frames = spec.shape[-1]
        if t_frames == 0:
            return spec
        max_w = min(self.max_width, t_frames)
        if max_w < 1:
            return spec
        w = int(torch.randint(1, max_w + 1, (1,)).item())
        t0 = int(torch.randint(0, t_frames - w + 1, (1,)).item())
        out = spec.clone()
        out[:, :, :, t0 : t0 + w] = 0.0
        return out


class Mixup:
    def __init__(self, alpha: float) -> None:
        self.alpha = max(float(alpha), 1e-8)

    def __call__(self, spec_a: torch.Tensor, spec_b: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if spec_a.shape != spec_b.shape:
            raise ValueError("spec_a and spec_b must have the same shape")
        dist = torch.distributions.Beta(
            torch.tensor(self.alpha, device=spec_a.device, dtype=torch.float32),
            torch.tensor(self.alpha, device=spec_a.device, dtype=torch.float32),
        )
        batch = spec_a.shape[0]
        lam = dist.sample((batch,)).to(device=spec_a.device, dtype=spec_a.dtype)
        view_shape = (batch,) + (1,) * (spec_a.dim() - 1)
        lam_b = lam.view(view_shape)
        mixed = lam_b * spec_a + (1.0 - lam_b) * spec_b
        weights = torch.stack((lam, 1.0 - lam), dim=-1)
        return mixed, weights
