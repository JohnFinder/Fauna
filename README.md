# Fauna

Bidirectional Budgerigar contact-call translator using an intermediate semantic tag system.

Fauna encodes bird audio recordings into structured semantic tags that humans can read, and maps human intent back to matching calls in a library. The core encoder (**BudgiFormer**) is a Conformer-based transformer pre-trained with masked spectrogram modeling, then fine-tuned to predict a curated ontology of acoustic, functional, and contextual tags.

## How it works

```
Bird audio ──► Mel spectrogram ──► BudgiFormer encoder ──► Tag predictions ──► Human text
Human text ──► Tag parser ──► Tag set ──► Retrieval engine ──► Best-matching call
```

The intermediate tag layer makes both directions interpretable. Tags carry evidence levels (literature-supported, model-inferred) so the system never overclaims.

## Requirements

- Python 3.13+
- [uv](https://docs.astral.sh/uv/) package manager

## Setup

```bash
git clone <repo-url> && cd Fauna
uv sync              # install runtime dependencies
uv sync --group dev  # also install pytest + pytest-cov
```

## Quick start

Run the full pipeline with synthetic audio and an untrained model:

```bash
uv run python scripts/inference.py
```

With a real recording:

```bash
uv run python scripts/inference.py --audio_path path/to/call.wav
```

With a trained checkpoint:

```bash
uv run python scripts/inference.py \
    --audio_path call.wav \
    --checkpoint checkpoints/tagger.pt \
    --threshold 0.5
```

| Flag | Default | Description |
|---|---|---|
| `--audio_path` | None (synthetic sweep) | Path to a WAV file |
| `--checkpoint` | None (random weights) | Path to a `.pt` checkpoint |
| `--config` | None (BUDGIFORMER\_S) | Path to a YAML config override |
| `--threshold` | 0.5 | Min probability to include a tag |

## Training

### Stage 1 — Self-supervised pre-training (MAE)

The encoder learns general Budgerigar audio representations by masking 75% of spectrogram patches and reconstructing them.

```python
import yaml
from src.training.pretrain import MAETrainer

with open("config/default.yaml") as f:
    config = yaml.safe_load(f)

trainer = MAETrainer(config, device="cuda")

for epoch in range(config["pretrain"]["epochs"]):
    loss = trainer.train_epoch(train_loader)
    print(f"Epoch {epoch}: loss={loss:.4f}")

trainer.save_checkpoint("checkpoints/mae_pretrained.pt")
```

The `train_loader` should yield batches of log-mel spectrograms with shape `(batch, 1, 128, T)`, where `T` is divisible by `patch_time` (default 4).

### Stage 2 — Supervised fine-tuning

Fine-tune the pre-trained encoder to predict semantic tags.

```python
from src.training.finetune import TagFineTuner

tuner = TagFineTuner(
    config,
    pretrained_encoder_path="checkpoints/mae_pretrained.pt",
    device="cuda",
)

for epoch in range(config["finetune"]["epochs"]):
    train_metrics = tuner.train_epoch(train_loader)
    val_metrics = tuner.evaluate(val_loader)
    print(f"Epoch {epoch}: F1={train_metrics['multilabel_f1']:.3f}")

tuner.save_checkpoint("checkpoints/tagger.pt")
```

The `train_loader` should yield `(spectrogram, tag_vector, context_features)` tuples.

## Dataset format

Provide a JSON manifest — a list of entries:

```json
[
  {
    "audio_path": "data/wav/call_001.wav",
    "caller_id": "bird_A",
    "context_tags": ["separation", "visiblePartner"],
    "functional_tags": ["contactCall", "flockLocalization"],
    "acoustic_tags": ["narrowbandFm"],
    "session_id": "session_1"
  }
]
```

Load it as a PyTorch dataset:

```python
from src.data.dataset import ContactCallDataset, ContactCallBatch
from torch.utils.data import DataLoader

ds = ContactCallDataset(manifest_path="data/calls.json")
loader = DataLoader(ds, batch_size=32, collate_fn=ContactCallBatch.collate)
```

For development without real data, generate a synthetic corpus:

```python
ds = ContactCallDataset.create_dummy(num_samples=500, num_callers=10)
```

## Text-to-bird (reverse direction)

Parse a human intent into ontology tags, then retrieve the closest call:

```python
from src.tags.text_to_tags import TextToTagsParser
from src.tags.retrieval import CallRetriever, CallLibraryEntry

parser = TextToTagsParser()
result = parser.parse("where are you")
# result.tags → ['contactCall', 'flockLocalization', 'socialContactMaintenance']

retriever = CallRetriever()
retriever.add_entry(CallLibraryEntry("calls/01.wav", {"contactCall", "flockLocalization"}))
matches = retriever.retrieve(result.tags, top_k=5)
```

## Tag ontology

Fauna v1.0.0 defines 19 tags across three layers:

| Layer | Tags | Evidence |
|---|---|---|
| **Acoustic** | `individualSignaturePresent`, `highModulation`, `narrowbandFm` | Supported |
| **Functional** | `contactCall`, `socialContactMaintenance`, `flockLocalization`, `sharedCallMatch`, `unknownFunction` | Supported |
| **Functional** | `socialApproachLikely`, `highArousalLikely` | Inferred |
| **Context** | `isolationOrSeparationContext`, `visiblePartner`, `audioOnlyPartner`, `separation`, `groupFlight`, `perchedSocial`, `operantTask`, `playbackTrial` | Supported |
| **Context** | `pairBondContextLikely` | Inferred |

**Supported** tags are grounded in published Budgerigar vocal-learning literature. **Inferred** tags are model predictions without direct playback validation. The system flags novel patterns that fall outside the learned distribution.

## Model architecture

**BudgiFormer** is a Conformer encoder operating on spectrogram patch tokens:

1. **Signal front-end** — 48 kHz, 5 ms window, 2.5 ms hop, 128 mel bins (500 Hz–10 kHz), pre-emphasis at 0.97
2. **Conv front-end** — two Conv2d layers (GELU) to lift the single-channel spectrogram into `dim` channels
3. **Patch embedding** — non-overlapping 16×4 patches projected to `dim`, with learnable 2D positional encoding and a prepended CLS token
4. **Encoder stack** — Conformer blocks (macaron-style half-step FFN sandwiching MHSA with relative sinusoidal positions and a depthwise conv module)
5. **Tag inference head** — prototypical pooling over patches for acoustic/functional tags, pooled MLP for context tags, with temperature-calibrated probabilities and a novelty detection flag

Two preset sizes:

| Variant | Layers | Dim | Heads | Params |
|---|---|---|---|---|
| BUDGIFORMER\_S | 8 | 256 | 4 | ~8M |
| BUDGIFORMER\_M | 12 | 384 | 6 | ~25M |

Switch between them by setting `model.size` to `small` or `medium` in `config/default.yaml`.

## Project structure

```
Fauna/
├── config/
│   └── default.yaml          # all hyperparameters
├── scripts/
│   └── inference.py           # end-to-end demo / inference CLI
├── src/
│   ├── data/
│   │   ├── augmentation.py    # TimeShift, GainJitter, FrequencyMask, TimeMask, Mixup
│   │   ├── dataset.py         # ContactCallDataset, collate, create_dummy
│   │   └── frontend.py        # MelSpectrogramFrontend, pad_or_trim
│   ├── model/
│   │   ├── budgiformer.py     # BudgiFormer encoder, configs, presets
│   │   ├── conformer.py       # Conformer blocks (FFN, MHSA, ConvModule)
│   │   ├── mae.py             # Masked autoencoder pre-training wrapper
│   │   ├── patch_embed.py     # ConvFrontend, PatchEmbedding, 2D PE, CLS
│   │   └── tag_head.py        # PrototypicalPooling, ContextMLP, TemperatureScaling
│   ├── tags/
│   │   ├── ontology.py        # Tag registry, layers, evidence levels
│   │   ├── retrieval.py       # Jaccard-based call retrieval
│   │   ├── tags_to_human.py   # Render tag predictions as human text
│   │   └── text_to_tags.py    # Parse human text into ontology tags
│   └── training/
│       ├── pretrain.py        # MAETrainer (self-supervised)
│       └── finetune.py        # TagFineTuner (supervised)
├── tests/                     # 196 pytest tests
├── pyproject.toml
└── uv.lock
```

## Tests

```bash
uv run pytest tests/ -v              # run all 196 tests
uv run pytest tests/ -v --cov src    # with coverage
```

## Configuration

All hyperparameters are centralized in `config/default.yaml`:

| Section | Key parameters |
|---|---|
| `data` | `sample_rate`, `max_duration_ms`, `n_mels`, `f_min`/`f_max` |
| `model` | `size`, `dim`, `num_layers`, `num_heads`, `dropout` |
| `mae` | `mask_ratio`, `decoder_dim`, `decoder_layers` |
| `pretrain` | `epochs`, `batch_size`, `lr`, `warmup_epochs`, `mixup_alpha` |
| `finetune` | `epochs`, `encoder_lr`, `head_lr`, `num_prototypes`, `novelty_threshold` |

## References

This system draws on:

- **Conformer** — Gulati et al. (2020), *Conformer: Convolution-augmented Transformer for Speech Recognition*
- **Bird-MAE** — Moummad & Farinas (2023), *Can Masked Autoencoders Also Listen to Birds?*
- **TweetyBERT** — Cohen et al. (2025), *Automated parsing of birdsong through self-supervised machine learning*
- **Budgerigar vocal learning** — Tu et al. (2011), *Learned vocalizations in budgerigars*; Banta Lavenex (1999), *Acoustic and perceptual categories of vocal elements in budgerigar warble song*

## License

MIT
