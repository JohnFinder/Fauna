from __future__ import annotations

import torch

from src.model.budgiformer import BudgiFormer, BudgiFormerConfig
from src.model.mae import BudgiFormerMAE, MAEDecoder
from src.model.tag_head import (
    ContextMLP,
    PrototypicalPooling,
    TagInferenceHead,
    TemperatureScaling,
)


def test_prototypical_pooling_output_shapes() -> None:
    torch.manual_seed(42)
    dim, num_tags, b, p = 32, 4, 2, 17
    pool = PrototypicalPooling(dim=dim, num_tags=num_tags, num_prototypes_per_tag=3)
    patch_embeddings = torch.randn(b, p, dim)
    logits, max_sim = pool(patch_embeddings)
    assert logits.shape == (b, num_tags)
    assert max_sim.shape == (b, num_tags)


def test_prototypical_pooling_similarity_range() -> None:
    torch.manual_seed(42)
    pool = PrototypicalPooling(dim=48, num_tags=3, num_prototypes_per_tag=5)
    patch_embeddings = torch.randn(2, 10, 48)
    patch_n = torch.nn.functional.normalize(patch_embeddings, dim=-1)
    proto_n = torch.nn.functional.normalize(pool.prototypes, dim=-1)
    sim = torch.einsum("bpd,tkd->btkp", patch_n, proto_n)
    s = sim.detach()
    assert float(s.min()) >= -1.0 - 1e-5
    assert float(s.max()) <= 1.0 + 1e-5


def test_context_mlp_output_shape() -> None:
    torch.manual_seed(42)
    dim, n_ctx_feat, n_ctx_tags = 32, 5, 7
    mlp = ContextMLP(dim=dim, num_context_features=n_ctx_feat, num_context_tags=n_ctx_tags)
    cls_e = torch.randn(2, dim)
    ctx = torch.randn(2, n_ctx_feat)
    y = mlp(cls_e, ctx)
    assert y.shape == (2, n_ctx_tags)


def test_tag_inference_head_forward_keys() -> None:
    torch.manual_seed(42)
    head = TagInferenceHead(
        dim=32,
        acoustic_functional_tags=["a", "b"],
        context_tags=["c"],
        num_context_features=4,
        num_prototypes=3,
    )
    pe = torch.randn(2, 8, 32)
    cls_e = torch.randn(2, 32)
    out = head(pe, cls_e, torch.zeros(2, 4))
    assert set(out.keys()) == {
        "acoustic_functional_logits",
        "context_logits",
        "max_similarities",
        "novelty_flag",
    }


def test_tag_inference_head_novelty_flag_dtype() -> None:
    torch.manual_seed(42)
    head = TagInferenceHead(
        dim=32,
        acoustic_functional_tags=["x"],
        context_tags=["y"],
        num_context_features=2,
    )
    out = head(torch.randn(2, 5, 32), torch.randn(2, 32))
    assert out["novelty_flag"].dtype == torch.bool


def test_tag_inference_head_no_context_features() -> None:
    torch.manual_seed(42)
    head = TagInferenceHead(
        dim=32,
        acoustic_functional_tags=["a", "b"],
        context_tags=["c"],
        num_context_features=3,
    )
    pe = torch.randn(2, 6, 32)
    cls_e = torch.randn(2, 32)
    out = head(pe, cls_e, context_features=None)
    assert out["context_logits"].shape == (2, 1)
    assert out["acoustic_functional_logits"].shape == (2, 2)


def test_predict_returns_probabilities() -> None:
    torch.manual_seed(42)
    head = TagInferenceHead(
        dim=32,
        acoustic_functional_tags=["a"],
        context_tags=["c1", "c2"],
        num_context_features=2,
    )
    pred = head.predict(torch.randn(2, 4, 32), torch.randn(2, 32))
    af = pred["acoustic_functional_probs"]
    ctx = pred["context_probs"]
    assert float(af.min()) >= 0.0 and float(af.max()) <= 1.0
    assert float(ctx.min()) >= 0.0 and float(ctx.max()) <= 1.0


def test_predict_temperature_effect() -> None:
    torch.manual_seed(42)
    head = TagInferenceHead(
        dim=32,
        acoustic_functional_tags=["a", "b"],
        context_tags=["c"],
        num_context_features=4,
    )
    pe = torch.randn(2, 6, 32)
    cls_e = torch.randn(2, 32) * 5.0
    ctx = torch.randn(2, 4) * 3.0
    out1 = head.predict(pe, cls_e, ctx, temperature=1.0)
    out2 = head.predict(pe, cls_e, ctx, temperature=50.0)
    d1 = (out1["context_probs"] - 0.5).abs().mean()
    d2 = (out2["context_probs"] - 0.5).abs().mean()
    assert float(d2) < float(d1)


def test_temperature_scaling_module() -> None:
    torch.manual_seed(42)
    ts = TemperatureScaling()
    with torch.no_grad():
        ts.temperature.fill_(2.0)
    logits = torch.randn(2, 5)
    y = ts(logits)
    assert torch.allclose(y, logits / 2.0)


def test_temperature_scaling_initial_value() -> None:
    torch.manual_seed(42)
    ts = TemperatureScaling()
    logits = torch.randn(3, 4)
    y = ts(logits)
    assert torch.allclose(y, logits)


def test_random_masking_shapes() -> None:
    torch.manual_seed(42)
    cfg = BudgiFormerConfig(
        n_mels=128,
        patch_freq=16,
        patch_time=4,
        dim=32,
        num_layers=1,
        num_heads=2,
        ffn_dim=64,
        conv_kernel_size=3,
        dropout=0.0,
        max_freq_patches=8,
        max_time_patches=64,
    )
    enc = BudgiFormer(cfg)
    dec = MAEDecoder(
        encoder_dim=cfg.dim,
        decoder_dim=32,
        num_layers=1,
        num_heads=2,
        patch_freq=cfg.patch_freq,
        patch_time=cfg.patch_time,
        max_num_patches=512,
        dropout=0.0,
    )
    mae = BudgiFormerMAE(enc, dec, mask_ratio=0.75)
    b, n, d = 2, 128, 32
    seq = torch.randn(b, n, d)
    unmasked, mask, ids_restore = mae.random_masking(seq, 0.75)
    len_keep = max(1, int(n * 0.25))
    assert unmasked.shape == (b, len_keep, d)
    assert mask.shape == (b, n)
    assert ids_restore.shape == (b, n)


def test_random_masking_ratio() -> None:
    torch.manual_seed(42)
    cfg = BudgiFormerConfig(
        n_mels=128,
        patch_freq=16,
        patch_time=4,
        dim=32,
        num_layers=1,
        num_heads=2,
        ffn_dim=64,
        conv_kernel_size=3,
        dropout=0.0,
        max_freq_patches=8,
        max_time_patches=64,
    )
    enc = BudgiFormer(cfg)
    dec = MAEDecoder(
        encoder_dim=cfg.dim,
        decoder_dim=32,
        num_layers=1,
        num_heads=2,
        max_num_patches=512,
        dropout=0.0,
    )
    mae = BudgiFormerMAE(enc, dec, mask_ratio=0.75)
    n = 200
    seq = torch.randn(4, n, 32)
    _, mask, _ = mae.random_masking(seq, 0.75)
    ratio = mask.mean().item()
    assert 0.65 < ratio < 0.85


def test_random_masking_mask_values() -> None:
    torch.manual_seed(42)
    cfg = BudgiFormerConfig(
        n_mels=128,
        patch_freq=16,
        patch_time=4,
        dim=32,
        num_layers=1,
        num_heads=2,
        ffn_dim=64,
        conv_kernel_size=3,
        dropout=0.0,
        max_freq_patches=8,
        max_time_patches=64,
    )
    enc = BudgiFormer(cfg)
    dec = MAEDecoder(
        encoder_dim=cfg.dim,
        decoder_dim=32,
        num_layers=1,
        num_heads=2,
        max_num_patches=512,
        dropout=0.0,
    )
    mae = BudgiFormerMAE(enc, dec)
    _, mask, _ = mae.random_masking(torch.randn(2, 50, 32), 0.5)
    u = mask.unique()
    assert len(u) == 2
    assert set(float(x) for x in u.tolist()) == {0.0, 1.0}


def test_random_masking_restore_is_permutation() -> None:
    torch.manual_seed(42)
    cfg = BudgiFormerConfig(
        n_mels=128,
        patch_freq=16,
        patch_time=4,
        dim=32,
        num_layers=1,
        num_heads=2,
        ffn_dim=64,
        conv_kernel_size=3,
        dropout=0.0,
        max_freq_patches=8,
        max_time_patches=64,
    )
    enc = BudgiFormer(cfg)
    dec = MAEDecoder(
        encoder_dim=cfg.dim,
        decoder_dim=32,
        num_layers=1,
        num_heads=2,
        max_num_patches=512,
        dropout=0.0,
    )
    mae = BudgiFormerMAE(enc, dec)
    b, n = 3, 37
    _, _, ids_restore = mae.random_masking(torch.randn(b, n, 16), 0.3)
    arange = torch.arange(n, device=ids_restore.device, dtype=ids_restore.dtype).unsqueeze(0).expand(b, -1)
    sorted_ids, _ = torch.sort(ids_restore, dim=1)
    assert torch.equal(sorted_ids, arange)


def test_mae_forward_loss_is_scalar() -> None:
    torch.manual_seed(42)
    cfg = BudgiFormerConfig(
        n_mels=128,
        patch_freq=16,
        patch_time=4,
        dim=32,
        num_layers=1,
        num_heads=2,
        ffn_dim=64,
        conv_kernel_size=3,
        dropout=0.0,
        max_freq_patches=8,
        max_time_patches=64,
    )
    enc = BudgiFormer(cfg)
    dec = MAEDecoder(
        encoder_dim=cfg.dim,
        decoder_dim=32,
        num_layers=1,
        num_heads=2,
        patch_freq=cfg.patch_freq,
        patch_time=cfg.patch_time,
        max_num_patches=512,
        dropout=0.0,
    )
    mae = BudgiFormerMAE(enc, dec, mask_ratio=0.75)
    spec = torch.randn(2, 1, 128, 64)
    loss, _, _ = mae(spec)
    assert loss.ndim == 0


def test_mae_forward_loss_is_finite() -> None:
    torch.manual_seed(42)
    cfg = BudgiFormerConfig(
        n_mels=128,
        patch_freq=16,
        patch_time=4,
        dim=32,
        num_layers=1,
        num_heads=2,
        ffn_dim=64,
        conv_kernel_size=3,
        dropout=0.0,
        max_freq_patches=8,
        max_time_patches=64,
    )
    enc = BudgiFormer(cfg)
    dec = MAEDecoder(
        encoder_dim=cfg.dim,
        decoder_dim=32,
        num_layers=1,
        num_heads=2,
        patch_freq=cfg.patch_freq,
        patch_time=cfg.patch_time,
        max_num_patches=512,
        dropout=0.0,
    )
    mae = BudgiFormerMAE(enc, dec)
    loss, _, _ = mae(torch.randn(2, 1, 128, 64))
    assert torch.isfinite(loss)


def test_mae_forward_reconstruction_shape() -> None:
    torch.manual_seed(42)
    cfg = BudgiFormerConfig(
        n_mels=128,
        patch_freq=16,
        patch_time=4,
        dim=32,
        num_layers=1,
        num_heads=2,
        ffn_dim=64,
        conv_kernel_size=3,
        dropout=0.0,
        max_freq_patches=8,
        max_time_patches=64,
    )
    enc = BudgiFormer(cfg)
    dec = MAEDecoder(
        encoder_dim=cfg.dim,
        decoder_dim=32,
        num_layers=1,
        num_heads=2,
        patch_freq=cfg.patch_freq,
        patch_time=cfg.patch_time,
        max_num_patches=512,
        dropout=0.0,
    )
    mae = BudgiFormerMAE(enc, dec)
    spec = torch.randn(2, 1, 128, 64)
    _, recon, _ = mae(spec)
    nf = cfg.n_mels // cfg.patch_freq
    nt = 64 // cfg.patch_time
    num_patches = nf * nt
    pred_dim = cfg.patch_freq * cfg.patch_time
    assert recon.shape == (2, num_patches, pred_dim)


def test_mae_forward_mask_shape() -> None:
    torch.manual_seed(42)
    cfg = BudgiFormerConfig(
        n_mels=128,
        patch_freq=16,
        patch_time=4,
        dim=32,
        num_layers=1,
        num_heads=2,
        ffn_dim=64,
        conv_kernel_size=3,
        dropout=0.0,
        max_freq_patches=8,
        max_time_patches=64,
    )
    enc = BudgiFormer(cfg)
    dec = MAEDecoder(
        encoder_dim=cfg.dim,
        decoder_dim=32,
        num_layers=1,
        num_heads=2,
        patch_freq=cfg.patch_freq,
        patch_time=cfg.patch_time,
        max_num_patches=512,
        dropout=0.0,
    )
    mae = BudgiFormerMAE(enc, dec)
    spec = torch.randn(2, 1, 128, 64)
    _, _, mask = mae(spec)
    nf = cfg.n_mels // cfg.patch_freq
    nt = 64 // cfg.patch_time
    assert mask.shape == (2, nf * nt)


def test_mae_is_differentiable() -> None:
    torch.manual_seed(42)
    cfg = BudgiFormerConfig(
        n_mels=128,
        patch_freq=16,
        patch_time=4,
        dim=32,
        num_layers=1,
        num_heads=2,
        ffn_dim=64,
        conv_kernel_size=3,
        dropout=0.0,
        max_freq_patches=8,
        max_time_patches=64,
    )
    enc = BudgiFormer(cfg)
    dec = MAEDecoder(
        encoder_dim=cfg.dim,
        decoder_dim=32,
        num_layers=1,
        num_heads=2,
        patch_freq=cfg.patch_freq,
        patch_time=cfg.patch_time,
        max_num_patches=512,
        dropout=0.0,
    )
    mae = BudgiFormerMAE(enc, dec)
    mae.train()
    spec = torch.randn(2, 1, 128, 64)
    loss, _, _ = mae(spec)
    loss.backward()
    assert loss.ndim == 0
