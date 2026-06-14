# Minecraft Schematic Retrieval (`mc-retrieval`)

This is the core codebase for the Minecraft Schematic Retrieval system. The system maps text queries (natural language) and 3D voxel representations of structures into a unified semantic space using a CLIP-style Dual-Encoder architecture.

---

## 📁 Repository Structure

```text
mc-retrieval/
├── configs/                  # Configuration YAML files (both CNN and Point-BERT baselines/experiments)
├── data/                     # Location for dataset parquet files
├── src/                      # Source Code
│   ├── dataset.py            # Preprocessing & loader for old (numeric) and new (voxel names) parquets
│   ├── model.py              # CNN-based dual-encoder models
│   ├── model_pointbert.py    # Point-BERT-based dual-encoder models
│   ├── train.py              # CNN training loop
│   ├── train_pointbert.py    # Point-BERT training loop
│   ├── evaluate.py           # Evaluation script for retrieval metrics (Recall@K, MRR)
│   ├── baselines.py          # TF-IDF, Random, and non-deep retrieval baselines
│   └── utils.py              # Utility helper functions
├── requirements.txt          # Python packages list
└── README_POINTBERT.md       # Point-BERT custom semantic enhancement strategies documentation
```

---

## 🚀 Getting Started

### 1. Installation
Install the required packages in your local environment or inside Google Colab:
```bash
pip install -r requirements.txt
```

### 2. Prepare Checkpoints & Data
- Place the Minecraft dataset parquet files inside the `data/` folder.
- If using Point-BERT, download the pretrained weights `Point-BERT.pth` and place them under `checkpoints/Point-BERT.pth`.

---

## 🧪 Experiments and Ablation Studies

We have implemented three semantic enhancement strategies under the Point-BERT setup to utilize human-readable voxel string names.

For a full breakdown of the strategies, config combinations, and commands to run these experiments, see the dedicated documentation:
* **[Point-BERT Semantics Enhancement Documentation](file:///C:/Users/farha/Desktop/semester%206/FP-IR/uni-ir/mc-retrieval/README_POINTBERT.md)**
