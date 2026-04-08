from __future__ import annotations

import json
import math
import tempfile
from pathlib import Path
from typing import Any, Callable, NamedTuple

import numpy as np
import soundfile as sf
import torch
import torchaudio

from src.data.frontend import pad_or_trim
from src.tags.ontology import TAG_REGISTRY, TagLayer, get_tags_by_layer


def _all_ontology_tag_names() -> list[str]:
    return sorted(TAG_REGISTRY.keys())


ALL_TAG_NAMES: list[str] = _all_ontology_tag_names()
TAG_TO_INDEX: dict[str, int] = {name: i for i, name in enumerate(ALL_TAG_NAMES)}
NUM_TAGS: int = len(ALL_TAG_NAMES)

CONTEXT_FEATURE_NAMES: list[str] = get_tags_by_layer(TagLayer.CONTEXT)
CONTEXT_TO_INDEX: dict[str, int] = {name: i for i, name in enumerate(CONTEXT_FEATURE_NAMES)}
NUM_CONTEXT_FEATURES: int = len(CONTEXT_FEATURE_NAMES)


class ContactCallBatch(NamedTuple):
    waveform: torch.Tensor
    tags: torch.Tensor
    context_features: torch.Tensor
    caller_id: list[str | None]
    session_id: list[str | None]

    @staticmethod
    def collate(batch: list[dict[str, Any]]) -> ContactCallBatch:
        waveforms = torch.stack([b["waveform"] for b in batch], dim=0)
        tags = torch.stack([b["tags"] for b in batch], dim=0)
        ctx = torch.stack([b["context_features"] for b in batch], dim=0)
        caller_ids = [b["caller_id"] for b in batch]
        session_ids = [b["session_id"] for b in batch]
        return ContactCallBatch(waveforms, tags, ctx, caller_ids, session_ids)


class ContactCallDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        manifest_path: str | None = None,
        entries: list[dict[str, Any]] | None = None,
        sample_rate: int = 48000,
        max_duration_ms: int = 400,
        transform: Callable[[torch.Tensor], torch.Tensor] | None = None,
    ) -> None:
        super().__init__()
        self.sample_rate = sample_rate
        self.max_duration_ms = max_duration_ms
        self.target_samples = int(round(sample_rate * (max_duration_ms / 1000.0)))
        self.transform = transform

        if manifest_path is not None and entries is not None:
            raise ValueError("Provide at most one of manifest_path or entries.")
        if manifest_path is not None:
            path = Path(manifest_path)
            raw = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(raw, list):
                raise ValueError("Manifest must be a JSON list of entries.")
            self._entries: list[dict[str, Any]] = list(raw)
        elif entries is not None:
            self._entries = list(entries)
        else:
            self._entries = []

    def __len__(self) -> int:
        return len(self._entries)

    def _encode_tags(self, entry: dict[str, Any]) -> torch.Tensor:
        vec = torch.zeros(NUM_TAGS, dtype=torch.float32)
        for key in ("context_tags", "functional_tags", "acoustic_tags"):
            for t in entry.get(key, []) or []:
                if t in TAG_TO_INDEX:
                    vec[TAG_TO_INDEX[t]] = 1.0
        return vec

    def _encode_context_features(self, entry: dict[str, Any]) -> torch.Tensor:
        vec = torch.zeros(NUM_CONTEXT_FEATURES, dtype=torch.float32)
        for t in entry.get("context_tags", []) or []:
            if t in CONTEXT_TO_INDEX:
                vec[CONTEXT_TO_INDEX[t]] = 1.0
        return vec

    def __getitem__(self, index: int) -> dict[str, Any]:
        entry = self._entries[index]
        path = entry["audio_path"]
        data, sr = sf.read(path, dtype="float32")
        if data.ndim == 1:
            wav = torch.from_numpy(data).unsqueeze(0)
        else:
            wav = torch.from_numpy(data.T)
        if wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)
        if sr != self.sample_rate:
            wav = torchaudio.functional.resample(wav, sr, self.sample_rate)
        wav = pad_or_trim(wav, self.target_samples, dim=-1)
        if self.transform is not None:
            wav = self.transform(wav)
        waveform = wav.squeeze(0).to(dtype=torch.float32)
        return {
            "waveform": waveform,
            "tags": self._encode_tags(entry),
            "context_features": self._encode_context_features(entry),
            "caller_id": entry.get("caller_id"),
            "session_id": entry.get("session_id"),
        }

    @classmethod
    def create_dummy(
        cls,
        num_samples: int = 100,
        num_callers: int = 5,
        sample_rate: int = 48000,
        max_duration_ms: int = 400,
        seed: int = 0,
    ) -> ContactCallDataset:
        rng = torch.Generator().manual_seed(seed)
        tmp_root = Path(tempfile.mkdtemp(prefix="fauna_dummy_calls_"))
        names = ALL_TAG_NAMES
        ctx_pool = [n for n in names if TAG_REGISTRY[n].layer == TagLayer.CONTEXT]
        func_pool = [n for n in names if TAG_REGISTRY[n].layer == TagLayer.FUNCTIONAL]
        aco_pool = [n for n in names if TAG_REGISTRY[n].layer == TagLayer.ACOUSTIC]

        target_samples = int(round(sample_rate * (max_duration_ms / 1000.0)))
        entries: list[dict[str, Any]] = []

        for i in range(num_samples):
            t = torch.arange(target_samples, dtype=torch.float32) / float(sample_rate)
            f0 = 800.0 + torch.rand((), generator=rng).item() * 1200.0
            f1 = f0 + 400.0 + torch.rand((), generator=rng).item() * 800.0
            phase = 2.0 * math.pi * (f0 + (f1 - f0) * t / t[-1].clamp_min(1e-6)) * t
            sweep = 0.25 * torch.sin(phase)
            noise = 0.05 * torch.randn(target_samples, generator=rng)
            mono = (sweep + noise).unsqueeze(0)

            out_path = tmp_root / f"dummy_{i:05d}.wav"
            sf.write(str(out_path), mono.squeeze(0).numpy(), sample_rate)

            k_ctx = int(torch.randint(0, 3, (1,), generator=rng).item())
            k_fn = int(torch.randint(1, 4, (1,), generator=rng).item())
            k_ac = int(torch.randint(0, 3, (1,), generator=rng).item())

            def pick(pool: list[str], k: int) -> list[str]:
                if not pool or k <= 0:
                    return []
                perm = torch.randperm(len(pool), generator=rng).tolist()
                return [pool[j] for j in perm[:k]]

            entries.append(
                {
                    "audio_path": str(out_path),
                    "caller_id": f"caller_{i % num_callers}",
                    "context_tags": pick(ctx_pool, k_ctx),
                    "functional_tags": pick(func_pool, k_fn),
                    "acoustic_tags": pick(aco_pool, k_ac),
                    "session_id": f"session_{i // max(1, num_samples // 10)}",
                }
            )

        return cls(entries=entries, sample_rate=sample_rate, max_duration_ms=max_duration_ms)
