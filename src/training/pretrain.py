from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import yaml
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from src.model.budgiformer import (
    BUDGIFORMER_M,
    BUDGIFORMER_S,
    BudgiFormer,
    BudgiFormerConfig,
)
from src.model.mae import BudgiFormerMAE, MAEDecoder


def _mel_time_frames(data_cfg: dict[str, Any]) -> int:
    sr = int(data_cfg["sample_rate"])
    max_ms = int(data_cfg["max_duration_ms"])
    wl = max(int(sr * max_ms / 1000), 1)
    n_fft = int(data_cfg["n_fft"])
    hop = int(data_cfg["hop_length"])
    if wl < n_fft:
        return 1
    return (wl - n_fft) // hop + 1


def budgiformer_config_from_yaml(cfg: dict[str, Any]) -> BudgiFormerConfig:
    d = cfg["data"]
    m = cfg["model"]
    preset = BUDGIFORMER_M if str(m.get("size", "small")).lower() == "medium" else BUDGIFORMER_S
    n_mels = int(d["n_mels"])
    pf = int(m["patch_freq"])
    pt = int(m["patch_time"])
    max_freq = max(n_mels // pf, 1, preset.max_freq_patches)
    mel_t = _mel_time_frames(d)
    max_time = max((mel_t + pt - 1) // pt, 1, preset.max_time_patches)
    return BudgiFormerConfig(
        n_mels=n_mels,
        patch_freq=pf,
        patch_time=pt,
        dim=int(m["dim"]),
        num_layers=int(m["num_layers"]),
        num_heads=int(m["num_heads"]),
        ffn_dim=int(m["ffn_dim"]),
        conv_kernel_size=int(m["conv_kernel_size"]),
        dropout=float(m["dropout"]),
        max_freq_patches=max_freq,
        max_time_patches=max_time,
    )


def _max_num_patches(enc_cfg: BudgiFormerConfig) -> int:
    return enc_cfg.max_freq_patches * enc_cfg.max_time_patches + 32


class MAETrainer:
    def __init__(self, config: dict[str, Any], device: str = "cpu") -> None:
        self.config = config
        self.device = torch.device(device)
        enc_cfg = budgiformer_config_from_yaml(config)
        encoder = BudgiFormer(enc_cfg)
        mae_cfg = config["mae"]
        decoder = MAEDecoder(
            encoder_dim=enc_cfg.dim,
            decoder_dim=int(mae_cfg["decoder_dim"]),
            num_layers=int(mae_cfg["decoder_layers"]),
            num_heads=int(mae_cfg["decoder_heads"]),
            patch_freq=enc_cfg.patch_freq,
            patch_time=enc_cfg.patch_time,
            max_num_patches=_max_num_patches(enc_cfg),
            dropout=float(config["model"]["dropout"]),
        )
        self.model = BudgiFormerMAE(
            encoder=encoder,
            decoder=decoder,
            mask_ratio=float(mae_cfg["mask_ratio"]),
        ).to(self.device)
        pt_cfg = config["pretrain"]
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=float(pt_cfg["lr"]),
            weight_decay=float(pt_cfg["weight_decay"]),
        )
        total_epochs = int(pt_cfg["epochs"])
        warmup_epochs = int(pt_cfg["warmup_epochs"])
        warmup = LinearLR(
            self.optimizer,
            start_factor=1e-6,
            end_factor=1.0,
            total_iters=max(warmup_epochs, 1),
        )
        cosine = CosineAnnealingLR(
            self.optimizer,
            T_max=max(total_epochs - warmup_epochs, 1),
            eta_min=float(pt_cfg["lr"]) * 1e-3,
        )
        self.scheduler = SequentialLR(
            self.optimizer,
            schedulers=[warmup, cosine],
            milestones=[warmup_epochs],
        )
        self.mixup_alpha = float(pt_cfg.get("mixup_alpha", 0.0))

    def train_epoch(self, dataloader: DataLoader) -> float:
        self.model.train()
        losses: list[float] = []
        for batch in tqdm(dataloader, desc="pretrain"):
            spec = batch[0].to(self.device) if isinstance(batch, (list, tuple)) else batch.to(self.device)
            if self.mixup_alpha > 0.0 and spec.size(0) > 1:
                lam = torch.distributions.Beta(
                    self.mixup_alpha,
                    self.mixup_alpha,
                ).sample().to(spec.device)
                perm = torch.randperm(spec.size(0), device=spec.device)
                spec = lam * spec + (1.0 - lam) * spec[perm]
            self.optimizer.zero_grad(set_to_none=True)
            loss, _, _ = self.model(spec)
            loss.backward()
            self.optimizer.step()
            losses.append(float(loss.detach().cpu()))
        self.scheduler.step()
        return sum(losses) / max(len(losses), 1)

    def save_checkpoint(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "scheduler": self.scheduler.state_dict(),
            },
            path,
        )

    def load_checkpoint(self, path: str | Path) -> None:
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt["model"])
        if "optimizer" in ckpt:
            self.optimizer.load_state_dict(ckpt["optimizer"])
        if "scheduler" in ckpt:
            self.scheduler.load_state_dict(ckpt["scheduler"])


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    cfg_path = root / "config" / "default.yaml"
    with open(cfg_path, encoding="utf-8") as f:
        config: dict[str, Any] = yaml.safe_load(f)
    enc_cfg = budgiformer_config_from_yaml(config)
    pt = int(enc_cfg.patch_time)
    mel_w = _mel_time_frames(config["data"])
    mel_w = max((mel_w // pt) * pt, pt)
    n_mels = enc_cfg.n_mels
    bsz = int(config["pretrain"]["batch_size"])
    specs = torch.randn(bsz, 1, n_mels, mel_w)
    ds = TensorDataset(specs)
    dl = DataLoader(ds, batch_size=bsz, shuffle=True)
    trainer = MAETrainer(config, device="cpu")
    _ = trainer.train_epoch(dl)


if __name__ == "__main__":
    main()
