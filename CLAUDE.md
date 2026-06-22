# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

Minecraft Schematic Retrieval research project. Maps natural language queries and 3D voxel Minecraft structures into a shared semantic embedding space using a CLIP-style Dual Encoder. All active code is in `mc-retrieval/`; `cbir-thing/` is a separate, standalone CBIR experiment.

## Commands

All scripts run from within `mc-retrieval/`. Install dependencies first:

```bash
pip install -r mc-retrieval/requirements.txt
```

### Training
```bash
# CNN-based baseline
python src/train.py --config configs/default.yaml

# Point-BERT frozen backbone (Plan 2 — recommended starting point)
python src/train_pointbert.py --config configs/pointbert.yaml

# Point-BERT with semantic strategies
python src/train_pointbert.py --config configs/pb_s1s2_semantic_init.yaml
python src/train_pointbert.py --config configs/pb_s1s2s3_all.yaml

# Full fine-tune warm-started from Plan 2 checkpoint (Plan 1)
python src/train_pointbert.py --config configs/pointbert_finetune.yaml \
    --warmstart checkpoints/pointbert_plan2/best.pt

# MVM self-supervised pretraining (CNN path only)
python src/pretrain.py --config configs/default.yaml
```

### Evaluation
```bash
python src/evaluate.py --config configs/pb_s1s2s3_all.yaml \
    --checkpoint checkpoints/pb_s1s2s3_all/best_model.pth
```

### Baselines
```bash
python src/baselines.py --config configs/default.yaml
```

### Utilities
```bash
# Export text inputs to CSV for review
python src/export_texts.py --parquet data/data_with_voxel_names_multiview_image.parquet
python src/export_texts.py --config configs/pb_s1s2_semantic_init.yaml

# Interactive retrieval demo
python src/retrieval_demo.py --config configs/pb_s1s2s3_all.yaml \
    --checkpoint checkpoints/pb_s1s2s3_all/best_model.pth
```

## Architecture

### Shared Embedding Space
Both encoders project into a **256-dim L2-normalized** space. Training uses symmetric InfoNCE (`CLIPLoss` in `losses.py`) with a learnable temperature clamped to [0.01, 1.0].

### Text Encoder (`model.py:TextEncoder`)
`all-MiniLM-L6-v2` (frozen, 22M params) → two-layer projection head (384→256→256, GELU). Gradients only flow through the projection head.

### Voxel Encoder — Two Backends

**Point-BERT path** (`model_pointbert.py`) — active research focus:
1. `VoxelToPoints`: 32³ grid → 512 sparse non-air points. Random sampling during training; FPS (farthest point sampling) during eval.
2. `nn.Embedding(vocab_size, 64)`: block ID/name → 64-dim feature (trainable; optionally semantic-initialized via S2).
3. `input_projection` Linear+LN+GELU+Linear (67→384): concatenated `[xyz(3) ∥ block_feat(64)]` → transformer token.
4. `PointBERTTransformer`: 12-layer Transformer (384-dim, 6-head). **Critical detail**: positional embeddings are added at every layer (`x = block(x + pos)`), matching the original Point-BERT implementation exactly.
5. `output_head` Linear (384→256).

**Plan 2** (`freeze_backbone: true`): only ~653K params train (adapter + head). **Plan 1** (`freeze_backbone: false`): full fine-tune with discriminative LRs (adapter 3e-4 >> backbone 5e-6).

**CNN path** (`model.py:VoxelEncoder`) — baseline/MVM pretraining:
Block embedding → 3D Conv stack with BatchNorm/GELU/Dropout/MaxPool → AdaptiveAvgPool3d(1) → projection. Supports depthwise-separable convolutions and a learned stem.

### Semantic Enhancement Strategies (Point-BERT only)

| Config flag | Strategy | Effect |
|---|---|---|
| `data.use_name_vocab: true` | S1 | Vocab built from block name strings instead of numeric IDs |
| `model.use_semantic_init: true` | S2 | `block_embedding` initialized from MiniLM encodings of block names; cached under `checkpoints/cache/block_emb_{md5}.npy` |
| `data.use_material_context: true` | S3 | Top-K dominant block names appended to text: `"[Materials: spruce_log, oak_planks, ...]"` |

S2 requires S1 (`__index_to_name__` key in `block_mapping`) to know which name maps to which compact ID.

### Data Pipeline (`dataset.py`)
- Two parquet formats: `data.parquet` (numeric IDs in `voxel_data`) and `data_with_voxel_names_multiview_image.parquet` (string names in `voxel_name_data`; needed for S1–S3).
- 8,328 samples, split 80/10/10 (train/val/test) via fixed seed.
- Text: `title + subtitle + description + tags` concatenated, HTML-stripped. S3 appends a materials suffix.
- Voxel preprocessing: remap IDs → compact indices, optional bbox crop + resize to 32³ (scipy zoom), optional augmentation (random 90° horizontal rotation + block dropout).
- Bbox cropping is the single highest-impact preprocessing step (4.4× R@1 improvement in ablations, driven by ~12× increase in input density).

### Checkpoint Format
Saved by `utils.save_checkpoint` as a dict with keys: `epoch`, `model_state`, `criterion_state`, `optimizer_state`, `val_loss`, `block_mapping`, `cfg`. The `block_mapping` must be saved alongside the model because it encodes the vocabulary built from training data.

Point-BERT pretrained weights (`Point-BERT.pth`) must be downloaded separately from the [Point-BERT GitHub repo](https://github.com/Julie-tang00/Point-BERT) and placed at the path specified by `pointbert.pretrained_path` in the config. The checkpoint loader strips `module.` prefixes, extracts the `transformer_q` branch, and remaps keys to match `PointBERTTransformer`'s structure; the `encoder` (PointNet group encoder) is deliberately skipped since the project uses a custom `InputAdapter` instead.

### Key Config Fields
Configs in `mc-retrieval/configs/`:
- `data.parquet_path` — which parquet file to load
- `pointbert.pretrained_path` — path to `Point-BERT.pth`
- `pointbert.freeze_backbone` — `true` = Plan 2, `false` = Plan 1
- `training.checkpoint_dir` — where `best.pt` and `last.pt` are saved
- `training.use_amp` — mixed precision (only active on CUDA)

### Metrics
`evaluate.py` reports two levels of metrics on the **test set** (833 samples, fixed seed=42, never seen during training):

- **Instance-level**: query[i] must retrieve exactly gallery[i] — the paired sample. Rank computed against all 833 candidates.
- **Category-level**: a retrieved item is "relevant" if it shares the same `subtitle` column value as the query. 15 categories total (e.g. `"Land Structure Map"`, `"3D Art Map"`, `"Redstone Device Map"`). Category precision@k = fraction of top-k results that share the query's category.

`--checkpoint` defaults to `{training.checkpoint_dir}/best.pt` if omitted.

**Known results (test set, 833 samples):**

| Config | T→V R@1 | T→V R@10 | T→V MRR | Cat P@1 |
|---|---|---|---|---|
| CNN + crop + pretrain + aug | 4.20% | 25.21% | 0.1105 | 51.62% |
| Point-BERT S1+S2 (Plan 2) | **6.12%** | **30.73%** | **0.1421** | **51.26%** |
