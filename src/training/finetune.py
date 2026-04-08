from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from src.data.frontend import MelSpectrogramFrontend
from src.model.budgiformer import (
    BUDGIFORMER_M,
    BUDGIFORMER_S,
    BudgiFormer,
    BudgiFormerConfig,
)
from src.model.mae import BudgiFormerMAE, MAEDecoder
from src.model.tag_head import TagInferenceHead, TemperatureScaling
from src.training.pretrain import budgiformer_config_from_yaml


def _dummy_tag_names(prefix: str, n: int) -> list[str]:
    return [f"{prefix}_{i}" for i in range(n)]


def _concat_logits(head_out: dict[str, torch.Tensor]) -> torch.Tensor:
    return torch.cat(
        [head_out["acoustic_functional_logits"], head_out["context_logits"]],
        dim=1,
    )


def _multilabel_f1_numpy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(
        f1_score(y_true, y_pred, average="macro", zero_division=0),
    )


def _ece_per_label(probs: np.ndarray, targets: np.ndarray, n_bins: int = 10) -> float:
    eces: list[float] = []
    n, k = probs.shape
    for j in range(k):
        p = probs[:, j]
        t = targets[:, j]
        order = np.argsort(p)
        p_s = p[order]
        t_s = t[order]
        bin_size = max(n // n_bins, 1)
        ece_j = 0.0
        for b in range(n_bins):
            lo = b * bin_size
            hi = min((b + 1) * bin_size, n)
            if lo >= hi:
                break
            pb = p_s[lo:hi]
            tb = t_s[lo:hi]
            conf = float(pb.mean())
            acc = float(tb.mean())
            w = (hi - lo) / n
            ece_j += w * abs(conf - acc)
        eces.append(ece_j)
    return float(np.mean(eces)) if eces else 0.0


class TagFineTuner:
    def __init__(
        self,
        config: dict[str, Any],
        pretrained_encoder_path: str | None = None,
        device: str = "cpu",
    ) -> None:
        self.config = config
        self.device = torch.device(device)
        enc_cfg: BudgiFormerConfig = budgiformer_config_from_yaml(config)
        self.encoder = BudgiFormer(enc_cfg).to(self.device)
        tags_cfg = config["tags"]
        n_af = int(tags_cfg["num_acoustic_functional"])
        n_ctx = int(tags_cfg["num_context"])
        n_ctx_feat = int(tags_cfg["num_context_features"])
        ft_cfg = config["finetune"]
        self.head = TagInferenceHead(
            dim=enc_cfg.dim,
            acoustic_functional_tags=_dummy_tag_names("af", n_af),
            context_tags=_dummy_tag_names("ctx", n_ctx),
            num_context_features=n_ctx_feat,
            num_prototypes=int(ft_cfg["num_prototypes"]),
            novelty_threshold=float(ft_cfg["novelty_threshold"]),
        ).to(self.device)
        self.temp = TemperatureScaling().to(self.device)
        if pretrained_encoder_path is not None:
            ckpt = torch.load(
                Path(pretrained_encoder_path),
                map_location=self.device,
                weights_only=False,
            )
            state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
            enc_sd = {
                k[len("encoder.") :]: v
                for k, v in state.items()
                if k.startswith("encoder.")
            }
            if enc_sd:
                self.encoder.load_state_dict(enc_sd, strict=True)
        enc_params: list[torch.nn.Parameter] = []
        head_params: list[torch.nn.Parameter] = []
        for p in self.encoder.parameters():
            enc_params.append(p)
        for p in self.head.parameters():
            head_params.append(p)
        for p in self.temp.parameters():
            head_params.append(p)
        self.optimizer = torch.optim.AdamW(
            [
                {"params": enc_params, "lr": float(ft_cfg["encoder_lr"])},
                {"params": head_params, "lr": float(ft_cfg["head_lr"])},
            ],
            weight_decay=float(ft_cfg["weight_decay"]),
        )
        n_tags = n_af + n_ctx
        self.tag_pos_weight = torch.ones(n_tags, device=self.device)

    def _forward_batch(
        self,
        spectrogram: torch.Tensor,
        context_features: torch.Tensor | None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        tokens = self.encoder(spectrogram)
        cls = tokens[:, 0, :]
        patches = tokens[:, 1:, :]
        out = self.head(patches, cls, context_features)
        logits = _concat_logits(out)
        logits = self.temp(logits)
        return logits, out

    def train_epoch(self, dataloader: DataLoader) -> dict[str, float]:
        self.encoder.train()
        self.head.train()
        self.temp.train()
        losses: list[float] = []
        all_logits: list[torch.Tensor] = []
        all_targets: list[torch.Tensor] = []
        pos_weight = self.tag_pos_weight
        for batch in tqdm(dataloader, desc="finetune"):
            spec = batch[0].to(self.device)
            targets = batch[1].to(self.device)
            ctx = batch[2].to(self.device) if len(batch) > 2 else None
            self.optimizer.zero_grad(set_to_none=True)
            logits, _ = self._forward_batch(spec, ctx)
            loss = F.binary_cross_entropy_with_logits(
                logits,
                targets,
                pos_weight=pos_weight,
            )
            loss.backward()
            self.optimizer.step()
            losses.append(float(loss.detach().cpu()))
            all_logits.append(logits.detach().cpu())
            all_targets.append(targets.detach().cpu())
        y_score = torch.cat(all_logits, dim=0).numpy()
        y_true = torch.cat(all_targets, dim=0).numpy()
        y_pred = (1.0 / (1.0 + np.exp(-y_score)) > 0.5).astype(np.float64)
        return {
            "loss": sum(losses) / max(len(losses), 1),
            "multilabel_f1": _multilabel_f1_numpy(y_true, y_pred),
        }

    def evaluate(self, dataloader: DataLoader) -> dict[str, float]:
        self.encoder.eval()
        self.head.eval()
        self.temp.eval()
        losses: list[float] = []
        all_logits: list[torch.Tensor] = []
        all_targets: list[torch.Tensor] = []
        pos_weight = self.tag_pos_weight
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="eval"):
                spec = batch[0].to(self.device)
                targets = batch[1].to(self.device)
                ctx = batch[2].to(self.device) if len(batch) > 2 else None
                logits, _ = self._forward_batch(spec, ctx)
                loss = F.binary_cross_entropy_with_logits(
                    logits,
                    targets,
                    pos_weight=pos_weight,
                )
                losses.append(float(loss.cpu()))
                all_logits.append(logits.cpu())
                all_targets.append(targets.cpu())
        y_score = torch.cat(all_logits, dim=0).numpy()
        y_true = torch.cat(all_targets, dim=0).numpy()
        probs = 1.0 / (1.0 + np.exp(-y_score))
        y_pred = (probs > 0.5).astype(np.float64)
        return {
            "loss": sum(losses) / max(len(losses), 1),
            "multilabel_f1": _multilabel_f1_numpy(y_true, y_pred),
            "calibration_error": _ece_per_label(probs, y_true),
        }

    def save_checkpoint(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "encoder": self.encoder.state_dict(),
                "head": self.head.state_dict(),
                "temp": self.temp.state_dict(),
                "optimizer": self.optimizer.state_dict(),
            },
            path,
        )

    def load_checkpoint(self, path: str | Path) -> None:
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.encoder.load_state_dict(ckpt["encoder"])
        self.head.load_state_dict(ckpt["head"])
        if "temp" in ckpt:
            self.temp.load_state_dict(ckpt["temp"])
        if "optimizer" in ckpt:
            self.optimizer.load_state_dict(ckpt["optimizer"])


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    cfg_path = root / "config" / "default.yaml"
    with open(cfg_path, encoding="utf-8") as f:
        config: dict[str, Any] = yaml.safe_load(f)
    d = config["data"]
    tags_cfg = config["tags"]
    n_af = int(tags_cfg["num_acoustic_functional"])
    n_ctx = int(tags_cfg["num_context"])
    n_tags = n_af + n_ctx
    n_ctx_feat = int(tags_cfg["num_context_features"])
    bsz = int(config["finetune"]["batch_size"])
    sr = int(d["sample_rate"])
    max_ms = int(d["max_duration_ms"])
    wl = int(sr * max_ms / 1000)
    wave = torch.randn(bsz, 1, wl)
    targets = (torch.rand(bsz, n_tags) > 0.7).float()
    ctx = torch.randn(bsz, n_ctx_feat)
    frontend = MelSpectrogramFrontend(
        sample_rate=sr,
        n_fft=int(d["n_fft"]),
        hop_length=int(d["hop_length"]),
        n_mels=int(d["n_mels"]),
        f_min=float(d["f_min"]),
        f_max=float(d["f_max"]),
    )
    with torch.no_grad():
        mel = frontend(wave)
    ds = TensorDataset(mel, targets, ctx)
    dl = DataLoader(ds, batch_size=bsz, shuffle=True)
    tuner = TagFineTuner(config, pretrained_encoder_path=None, device="cpu")
    _ = tuner.train_epoch(dl)


if __name__ == "__main__":
    main()
