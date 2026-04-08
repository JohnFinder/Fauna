from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import torch
import torchaudio
import yaml

from src.data.dataset import NUM_CONTEXT_FEATURES
from src.data.frontend import MelSpectrogramFrontend, pad_or_trim
from src.model.budgiformer import BUDGIFORMER_S, BudgiFormer, BudgiFormerConfig
from src.model.tag_head import TagInferenceHead
from src.tags.ontology import TAG_REGISTRY, TagLayer, get_tags_by_layer
from src.tags.tags_to_human import HumanRenderer


def _config_fields() -> set[str]:
    return {f.name for f in BudgiFormerConfig.__dataclass_fields__.values()}


def load_budgiformer_config(path: str | None) -> BudgiFormerConfig:
    if path is None:
        return BUDGIFORMER_S
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("Config YAML must map to a dictionary.")
    fields = _config_fields()
    kwargs = {k: v for k, v in raw.items() if k in fields}
    return BudgiFormerConfig(**kwargs)


def build_tag_head(dim: int) -> TagInferenceHead:
    ac = get_tags_by_layer(TagLayer.ACOUSTIC)
    fn = get_tags_by_layer(TagLayer.FUNCTIONAL)
    af = sorted(ac + fn)
    ctx = sorted(get_tags_by_layer(TagLayer.CONTEXT))
    return TagInferenceHead(
        dim=dim,
        acoustic_functional_tags=af,
        context_tags=ctx,
        num_context_features=NUM_CONTEXT_FEATURES,
    )


def load_checkpoint(
    encoder: BudgiFormer,
    head: TagInferenceHead,
    checkpoint_path: str | None,
) -> None:
    if checkpoint_path is None:
        return
    blob = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(blob, dict):
        raise ValueError("Checkpoint must be a dict.")
    enc_sd = blob.get("encoder") or blob.get("encoder_state_dict") or blob.get("budgiformer")
    head_sd = blob.get("tag_head") or blob.get("tag_head_state_dict")
    if enc_sd is not None:
        encoder.load_state_dict(enc_sd, strict=False)
    if head_sd is not None:
        head.load_state_dict(head_sd, strict=False)


def load_waveform(audio_path: str, sample_rate: int, max_ms: int) -> torch.Tensor:
    wav, sr = torchaudio.load(audio_path)
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sr != sample_rate:
        wav = torchaudio.functional.resample(wav, sr, sample_rate)
    n = int(round(sample_rate * (max_ms / 1000.0)))
    wav = pad_or_trim(wav, n, dim=-1)
    return wav.squeeze(0)


def dummy_waveform(sample_rate: int, max_ms: int, seed: int = 0) -> torch.Tensor:
    rng = torch.Generator().manual_seed(seed)
    n = int(round(sample_rate * (max_ms / 1000.0)))
    t = torch.arange(n, dtype=torch.float32) / float(sample_rate)
    f0 = 900.0
    f1 = 1800.0
    phase = 2.0 * math.pi * (f0 + (f1 - f0) * t / t[-1].clamp_min(1e-6)) * t
    sweep = 0.2 * torch.sin(phase)
    noise = 0.05 * torch.randn(n, generator=rng)
    return (sweep + noise).to(dtype=torch.float32)


def predictions_for_renderer(
    head: TagInferenceHead,
    pred: dict[str, torch.Tensor],
    threshold: float,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    af_names = head.acoustic_functional_tag_names
    ctx_names = head.context_tag_names
    af_p = pred["acoustic_functional_probs"][0]
    ctx_p = pred["context_probs"][0]
    for i, name in enumerate(af_names):
        p = float(af_p[i].item())
        if p >= threshold:
            layer = TAG_REGISTRY[name].layer
            kind = "acoustic" if layer == TagLayer.ACOUSTIC else "functional"
            out.append({"tag": name, "confidence": p, "kind": kind})
    for i, name in enumerate(ctx_names):
        p = float(ctx_p[i].item())
        if p >= threshold:
            out.append({"tag": name, "confidence": p, "kind": "functional"})
    return out


def _pad_spec_time(spec: torch.Tensor, patch_time: int) -> torch.Tensor:
    t = spec.shape[-1]
    rem = t % patch_time
    if rem == 0:
        return spec
    return torch.nn.functional.pad(spec, (0, patch_time - rem), mode="constant", value=0.0)


def run_pipeline(
    waveform_1d: torch.Tensor,
    mel: MelSpectrogramFrontend,
    encoder: BudgiFormer,
    head: TagInferenceHead,
    device: torch.device,
    prob_threshold: float,
) -> tuple[str, dict[str, Any]]:
    encoder.eval()
    head.eval()
    mel.eval()
    w = waveform_1d.unsqueeze(0).to(device)
    with torch.no_grad():
        spec = mel(w)
        spec = _pad_spec_time(spec, encoder.config.patch_time)
        tokens = encoder(spec)
        cls_emb = tokens[:, 0, :]
        patches = tokens[:, 1:, :]
        pred = head.predict(patches, cls_emb, None)
        novelty = bool(pred["novelty_flag"][0].item())
        human_in = predictions_for_renderer(head, pred, prob_threshold)
        text = HumanRenderer().render(human_in, novelty)
        summary: dict[str, Any] = {
            "acoustic_functional_probs": pred["acoustic_functional_probs"][0].cpu().tolist(),
            "context_probs": pred["context_probs"][0].cpu().tolist(),
            "novelty": novelty,
            "af_names": head.acoustic_functional_tag_names,
            "ctx_names": head.context_tag_names,
        }
    return text, summary


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--audio_path", type=str, default=None)
    p.add_argument("--checkpoint", type=str, default=None)
    p.add_argument("--config", type=str, default=None)
    p.add_argument("--threshold", type=float, default=0.5)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = load_budgiformer_config(args.config)
    encoder = BudgiFormer(cfg).to(device)
    head = build_tag_head(cfg.dim).to(device)
    load_checkpoint(encoder, head, args.checkpoint)

    sample_rate = 48000
    max_ms = 400
    mel = MelSpectrogramFrontend(sample_rate=sample_rate, n_mels=cfg.n_mels).to(device)

    if args.audio_path is None:
        wave = dummy_waveform(sample_rate, max_ms)
    else:
        wave = load_waveform(args.audio_path, sample_rate, max_ms)

    text, summary = run_pipeline(wave, mel, encoder, head, device, args.threshold)

    print(text)
    print("novelty:", summary["novelty"])


if __name__ == "__main__":
    main()
