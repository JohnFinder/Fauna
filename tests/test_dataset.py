from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest
import soundfile as sf
import torch

from src.data.dataset import (
    ALL_TAG_NAMES,
    CONTEXT_FEATURE_NAMES,
    CONTEXT_TO_INDEX,
    NUM_CONTEXT_FEATURES,
    NUM_TAGS,
    TAG_TO_INDEX,
    ContactCallBatch,
    ContactCallDataset,
)
from src.tags.ontology import TAG_REGISTRY, TagLayer


def test_all_tag_names_matches_registry():
    assert set(ALL_TAG_NAMES) == set(TAG_REGISTRY.keys())


def test_tag_to_index_covers_all_tags():
    assert len(TAG_TO_INDEX) == NUM_TAGS
    assert set(TAG_TO_INDEX.keys()) == set(ALL_TAG_NAMES)


def test_tag_indices_are_contiguous():
    assert sorted(TAG_TO_INDEX.values()) == list(range(NUM_TAGS))


def test_context_feature_names_are_context_layer():
    for name in CONTEXT_FEATURE_NAMES:
        assert TAG_REGISTRY[name].layer == TagLayer.CONTEXT


def test_context_to_index_covers_all_context():
    assert len(CONTEXT_TO_INDEX) == NUM_CONTEXT_FEATURES
    assert set(CONTEXT_TO_INDEX.keys()) == set(CONTEXT_FEATURE_NAMES)


def test_num_context_features_positive():
    assert NUM_CONTEXT_FEATURES > 0


def _write_dummy_wav(path: Path, sr: int = 48000, duration_ms: int = 200) -> None:
    n = int(sr * duration_ms / 1000)
    wav = torch.randn(n).numpy()
    sf.write(str(path), wav, sr)


def _make_entry(audio_path: str) -> dict:
    return {
        "audio_path": audio_path,
        "caller_id": "bird_0",
        "context_tags": ["separation", "visiblePartner"],
        "functional_tags": ["contactCall", "flockLocalization"],
        "acoustic_tags": ["narrowbandFm"],
        "session_id": "sess_0",
    }


def test_dataset_from_entries():
    with tempfile.TemporaryDirectory() as tmpdir:
        p = Path(tmpdir) / "call.wav"
        _write_dummy_wav(p)
        entry = _make_entry(str(p))
        ds = ContactCallDataset(entries=[entry], sample_rate=48000, max_duration_ms=200)
        assert len(ds) == 1


def test_dataset_getitem_keys():
    with tempfile.TemporaryDirectory() as tmpdir:
        p = Path(tmpdir) / "call.wav"
        _write_dummy_wav(p)
        ds = ContactCallDataset(entries=[_make_entry(str(p))])
        item = ds[0]
        assert set(item.keys()) == {"waveform", "tags", "context_features", "caller_id", "session_id"}


def test_dataset_waveform_length():
    with tempfile.TemporaryDirectory() as tmpdir:
        p = Path(tmpdir) / "call.wav"
        _write_dummy_wav(p, duration_ms=300)
        ds = ContactCallDataset(entries=[_make_entry(str(p))], max_duration_ms=400)
        item = ds[0]
        expected = int(round(48000 * 0.4))
        assert item["waveform"].shape[0] == expected


def test_dataset_tag_encoding():
    with tempfile.TemporaryDirectory() as tmpdir:
        p = Path(tmpdir) / "call.wav"
        _write_dummy_wav(p)
        ds = ContactCallDataset(entries=[_make_entry(str(p))])
        item = ds[0]
        tags = item["tags"]
        assert tags.shape == (NUM_TAGS,)
        assert tags[TAG_TO_INDEX["contactCall"]] == 1.0
        assert tags[TAG_TO_INDEX["flockLocalization"]] == 1.0
        assert tags[TAG_TO_INDEX["narrowbandFm"]] == 1.0
        assert tags[TAG_TO_INDEX["separation"]] == 1.0
        assert tags[TAG_TO_INDEX["visiblePartner"]] == 1.0
        assert tags[TAG_TO_INDEX["highModulation"]] == 0.0


def test_dataset_context_features():
    with tempfile.TemporaryDirectory() as tmpdir:
        p = Path(tmpdir) / "call.wav"
        _write_dummy_wav(p)
        ds = ContactCallDataset(entries=[_make_entry(str(p))])
        item = ds[0]
        ctx = item["context_features"]
        assert ctx.shape == (NUM_CONTEXT_FEATURES,)
        assert ctx[CONTEXT_TO_INDEX["separation"]] == 1.0
        assert ctx[CONTEXT_TO_INDEX["visiblePartner"]] == 1.0
        assert ctx[CONTEXT_TO_INDEX["groupFlight"]] == 0.0


def test_dataset_caller_and_session():
    with tempfile.TemporaryDirectory() as tmpdir:
        p = Path(tmpdir) / "call.wav"
        _write_dummy_wav(p)
        ds = ContactCallDataset(entries=[_make_entry(str(p))])
        item = ds[0]
        assert item["caller_id"] == "bird_0"
        assert item["session_id"] == "sess_0"


def test_dataset_from_manifest():
    with tempfile.TemporaryDirectory() as tmpdir:
        wav_path = Path(tmpdir) / "call.wav"
        _write_dummy_wav(wav_path)
        manifest = Path(tmpdir) / "manifest.json"
        manifest.write_text(json.dumps([_make_entry(str(wav_path))]))
        ds = ContactCallDataset(manifest_path=str(manifest))
        assert len(ds) == 1
        item = ds[0]
        assert item["caller_id"] == "bird_0"


def test_dataset_both_sources_raises():
    with pytest.raises(ValueError, match="at most one"):
        ContactCallDataset(manifest_path="x.json", entries=[{}])


def test_dataset_empty():
    ds = ContactCallDataset()
    assert len(ds) == 0


def test_create_dummy_produces_samples():
    ds = ContactCallDataset.create_dummy(num_samples=10, num_callers=3)
    assert len(ds) == 10
    item = ds[0]
    assert item["waveform"].ndim == 1
    assert item["tags"].shape == (NUM_TAGS,)
    assert item["context_features"].shape == (NUM_CONTEXT_FEATURES,)


def test_create_dummy_deterministic():
    ds1 = ContactCallDataset.create_dummy(num_samples=5, seed=123)
    ds2 = ContactCallDataset.create_dummy(num_samples=5, seed=123)
    for i in range(5):
        assert torch.allclose(ds1[i]["waveform"], ds2[i]["waveform"])
        assert torch.equal(ds1[i]["tags"], ds2[i]["tags"])


def test_collate_batch():
    with tempfile.TemporaryDirectory() as tmpdir:
        paths = []
        for i in range(3):
            p = Path(tmpdir) / f"call_{i}.wav"
            _write_dummy_wav(p)
            paths.append(p)
        entries = [_make_entry(str(p)) for p in paths]
        ds = ContactCallDataset(entries=entries, max_duration_ms=200)
        items = [ds[i] for i in range(3)]
        batch = ContactCallBatch.collate(items)
        assert batch.waveform.shape[0] == 3
        assert batch.tags.shape == (3, NUM_TAGS)
        assert batch.context_features.shape == (3, NUM_CONTEXT_FEATURES)
        assert len(batch.caller_id) == 3
        assert len(batch.session_id) == 3


def test_collate_with_dataloader():
    ds = ContactCallDataset.create_dummy(num_samples=8, num_callers=2)
    loader = torch.utils.data.DataLoader(ds, batch_size=4, collate_fn=ContactCallBatch.collate)
    batch = next(iter(loader))
    assert isinstance(batch, ContactCallBatch)
    assert batch.waveform.shape[0] == 4


def test_unknown_tags_ignored():
    with tempfile.TemporaryDirectory() as tmpdir:
        p = Path(tmpdir) / "call.wav"
        _write_dummy_wav(p)
        entry = {
            "audio_path": str(p),
            "caller_id": "b",
            "context_tags": ["madeUpTag"],
            "functional_tags": ["contactCall"],
            "acoustic_tags": [],
            "session_id": "s",
        }
        ds = ContactCallDataset(entries=[entry])
        item = ds[0]
        assert item["tags"][TAG_TO_INDEX["contactCall"]] == 1.0
        assert item["tags"].sum() == 1.0
