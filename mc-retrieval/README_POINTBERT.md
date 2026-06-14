# Point-BERT Semantic Enhancements for Minecraft Schematic Retrieval

This section documents the **Point-BERT integration** and the **three semantic enhancement strategies** designed to leverage human-readable block name strings (`voxel_name_data` from `data_with_voxel_names_multiview_image.parquet`) instead of arbitrary, randomly-initialized block ID numbers.

---

## 💡 Semantic Enhancement Strategies

### 1. Strategy 1: Name-Based Block Vocabulary (`use_name_vocab: true`)
* **Core Concept**: Traditional mapping constructs vocabularies based on block frequency rankings. This strategy replaces that by constructing a vocabulary using human-readable block names (e.g., `'minecraft:spruce_log'`, `'minecraft:stone_bricks'`).
* **Why it matters**: Groups semantically similar blocks together and prepares the block vocabulary for semantic initialization using pre-trained language models.

### 2. Strategy 2: Semantic Embedding Initialization (`use_semantic_init: true`)
* **Core Concept**: Pre-computes block name text embeddings using the same `SentenceTransformer` text model (e.g., `all-MiniLM-L6-v2`) and projects them into the `block_embedding` dimension.
* **Why it matters**: Voxel features start with close embeddings if their block names are semantically related (e.g., `oak_log` vs. `spruce_log`), allowing the model to quickly leverage language concepts.
* **Smart Caching**: Uses a local cache system. It hashes the block vocabulary name list (using MD5) and saves/loads the pre-computed embeddings to/from `checkpoints/cache/block_emb_{hash}.npy`. This avoids running text embedding inference repeatedly on subsequent training runs.
* **Freeze Option**: Control whether the semantic weights are frozen or trainable during training using `semantic_init_freeze: true/false`.

### 3. Strategy 3: Material Context Text Augmentation (`use_material_context: true`)
* **Core Concept**: Automatically extracts the top-K dominant block/material names from the voxel grid and appends them to the schematic's text description (e.g., `[Materials: spruce_log, cobblestone, oak_planks]`).
* **Why it matters**: Directly exposes material compositions to the Text Encoder, allowing text-voxel alignment to leverage specific building materials (e.g., *"wooden cabin"* matches the augmented text).

---

## ⚙️ Configuration Setup

Four configuration files have been provided under the `configs/` directory:

| Config File | Strategy Active | Focus / Ablation |
|---|---|---|
| [`configs/pointbert.yaml`](file:///C:/Users/farha/Desktop/semester%206/FP-IR/uni-ir/mc-retrieval/configs/pointbert.yaml) | Baseline | Standard Point-BERT without modifications |
| [`configs/pb_s1_name_vocab.yaml`](file:///C:/Users/farha/Desktop/semester%206/FP-IR/uni-ir/mc-retrieval/configs/pb_s1_name_vocab.yaml) | **Strategy 1** | Name-based Vocabulary only |
| [`configs/pb_s1s2_semantic_init.yaml`](file:///C:/Users/farha/Desktop/semester%206/FP-IR/uni-ir/mc-retrieval/configs/pb_s1s2_semantic_init.yaml) | **Strategy 1 & 2** | Semantic Initialization of Embeddings |
| [`configs/pb_s3_material_ctx.yaml`](file:///C:/Users/farha/Desktop/semester%206/FP-IR/uni-ir/mc-retrieval/configs/pb_s3_material_ctx.yaml) | **Strategy 3** | Text Augmentation with Dominant Materials |
| [`configs/pb_s1s2s3_all.yaml`](file:///C:/Users/farha/Desktop/semester%206/FP-IR/uni-ir/mc-retrieval/configs/pb_s1s2s3_all.yaml) | **All Strategies** | Full Ablation (S1 + S2 + S3) |

---

## 🚀 Running Experiments

### Prerequisites
Make sure your dataset contains the `voxel_name_data` column (e.g., `data_with_voxel_names_multiview_image.parquet`) and the official pretrained checkpoint is located at `checkpoints/Point-BERT.pth`.

### Training
Run any of the experiments by specifying the config path:

```bash
# Run Strategy 1 + 2
python src/train_pointbert.py --config configs/pb_s1s2_semantic_init.yaml

# Run All Strategies combined
python src/train_pointbert.py --config configs/pb_s1s2s3_all.yaml
```

*(On Google Colab, prepend with `!`)*

### Evaluation
Evaluate a trained model checkpoint:
```bash
python src/evaluate.py --config configs/pb_s1s2s3_all.yaml --checkpoint checkpoints/pb_s1s2s3_all/best_model.pth
```
