from __future__ import annotations

import torch

from src.data.frontend import MelSpectrogramFrontend, _preemphasis
from src.model.budgiformer import BUDGIFORMER_S, BudgiFormer, BudgiFormerConfig
from src.model.conformer import ConformerBlock
from src.model.mae import BudgiFormerMAE, MAEDecoder
from src.model.tag_head import TagInferenceHead
from src.tags.retrieval import CallLibraryEntry, CallRetriever
from src.tags.tags_to_human import HumanRenderer
from src.tags.text_to_tags import TextToTagsParser


def test_mel_frontend() -> None:
    fe = MelSpectrogramFrontend(sample_rate=48000, n_fft=240, hop_length=120, n_mels=128)
    wave = torch.randn(2, 3200)
    x = _preemphasis(wave.unsqueeze(1), fe.preemph_coeff)
    ref = fe.mel(x.squeeze(1))
    out = fe(wave)
    assert out.shape == (2, 1, 128, ref.shape[-1])


def test_conformer_block() -> None:
    blk = ConformerBlock(dim=32, num_heads=2, ffn_dim=64, conv_kernel_size=3, dropout=0.0)
    x = torch.randn(2, 12, 32)
    y = blk(x)
    assert y.shape == x.shape


def test_budgiformer_forward() -> None:
    m = BudgiFormer(BUDGIFORMER_S)
    spec = torch.randn(2, 1, 128, 256)
    y = m(spec)
    nf = 128 // BUDGIFORMER_S.patch_freq
    nt = 256 // BUDGIFORMER_S.patch_time
    assert y.shape == (2, nf * nt + 1, BUDGIFORMER_S.dim)


def test_tag_head() -> None:
    af = ["a0", "a1"]
    ctx = ["c0"]
    head = TagInferenceHead(
        dim=48,
        acoustic_functional_tags=af,
        context_tags=ctx,
        num_context_features=3,
        num_prototypes=2,
        novelty_threshold=0.5,
    )
    pe = torch.randn(2, 5, 48)
    cls_e = torch.randn(2, 48)
    ctx_f = torch.zeros(2, 3)
    out = head(pe, cls_e, ctx_f)
    assert out["acoustic_functional_logits"].shape == (2, len(af))
    assert out["context_logits"].shape == (2, len(ctx))
    assert out["novelty_flag"].shape == (2,)
    assert out["novelty_flag"].dtype == torch.bool


def test_mae_forward() -> None:
    cfg = BudgiFormerConfig(
        n_mels=128,
        patch_freq=16,
        patch_time=4,
        dim=64,
        num_layers=1,
        num_heads=2,
        ffn_dim=128,
        conv_kernel_size=3,
        dropout=0.0,
        max_freq_patches=8,
        max_time_patches=32,
    )
    enc = BudgiFormer(cfg)
    dec = MAEDecoder(
        encoder_dim=cfg.dim,
        decoder_dim=64,
        num_layers=1,
        num_heads=2,
        patch_freq=cfg.patch_freq,
        patch_time=cfg.patch_time,
        max_num_patches=512,
        dropout=0.0,
    )
    mae = BudgiFormerMAE(enc, dec, mask_ratio=0.5)
    spec = torch.randn(2, 1, 128, 64)
    loss, recon, mask = mae(spec)
    assert loss.ndim == 0
    nf = 8
    nt = 16
    n_p = nf * nt
    assert recon.shape == (2, n_p, cfg.patch_freq * cfg.patch_time)
    assert mask.shape == (2, n_p)


def test_text_to_tags() -> None:
    p = TextToTagsParser()
    r1 = p.parse("where are you")
    assert not r1.rejected
    assert set(r1.tags) >= {"contactCall", "socialContactMaintenance", "flockLocalization"}
    r2 = p.parse("predator attack")
    assert r2.rejected


def test_human_renderer() -> None:
    r = HumanRenderer()
    text = r.render([{"tag": "contactCall", "confidence": 0.9, "kind": "functional"}], False)
    assert isinstance(text, str) and len(text) > 0


def test_retrieval() -> None:
    ret = CallRetriever()
    ret.add_entry(CallLibraryEntry("a.wav", {"contactCall", "flockLocalization"}))
    ret.add_entry(CallLibraryEntry("b.wav", {"playbackTrial"}))
    ret.add_entry(CallLibraryEntry("c.wav", {"contactCall", "flockLocalization", "playbackTrial"}))
    ranked = ret.retrieve(["contactCall", "flockLocalization"], top_k=3)
    assert [e.audio_path for e, _ in ranked] == ["a.wav", "c.wav", "b.wav"]
    assert ranked[0][1] >= ranked[1][1] >= ranked[2][1]
