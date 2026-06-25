"""Dataset for trimodal (text + image + voxel) retrieval."""

import os
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from transformers import CLIPProcessor
from sklearn.model_selection import train_test_split

from dataset import (
    build_block_name_mapping,
    remap_voxel_from_names,
    augment_voxel,
    build_text,
    build_text_with_materials,
)


def resolve_image_path(
    row: pd.Series,
    renders_base: str,
    view_idx: int,
    n_views: int = 12,
) -> Optional[str]:
    """Resolve rendered image path from a parquet row.

    Priority:
    1. view_XX_path column (absolute or relative via renders_base)
    2. render_folder column + view_XX.jpg filename
    """
    col = f"view_{view_idx:02d}_path"
    stored = row.get(col)
    if isinstance(stored, str) and stored:
        if os.path.exists(stored):
            return stored
        parts = Path(stored).parts
        if len(parts) >= 2:
            candidate = os.path.join(renders_base, parts[-2], parts[-1])
            if os.path.exists(candidate):
                return candidate

    render_folder = row.get("render_folder")
    if isinstance(render_folder, str):
        candidate = os.path.join(renders_base, render_folder, f"view_{view_idx:02d}.jpg")
        if os.path.exists(candidate):
            return candidate

    return None


class TriModalDataset(Dataset):
    """Each item: (text, pixel_values, voxel, category).

    Image selection:
      train → random view from [0, n_views)
      val/test → fixed eval_view_idx (default 6)
    Missing images fall back to a black RGB image.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        block_mapping: dict,
        processor: CLIPProcessor,
        cfg: dict,
        split: str = "train",
    ):
        data_cfg = cfg["data"]
        self.df = df.reset_index(drop=True)
        self.block_mapping = block_mapping
        self.processor = processor
        self.renders_base = data_cfg.get("renders_base", "data/renders")
        self.n_views = data_cfg.get("n_views", 12)
        self.n_views_use = data_cfg.get("n_views_use", 6)
        self.image_size = cfg["model"].get("image_size", 224)
        self.is_train = split == "train"
        self.crop_bbox = data_cfg.get("crop_bbox", True)
        self.aug = data_cfg.get("augment", False) and self.is_train
        self.aug_apply_prob = data_cfg.get("aug_apply_prob", 0.5)
        self.aug_dropout_prob = data_cfg.get("aug_dropout_prob", 0.05)
        self.fallback_img = Image.new("RGB", (self.image_size, self.image_size), (0, 0, 0))
        # fixed evenly-spaced view indices (same for train and eval, like TriCoLo)
        self._view_indices = [
            round(i * (self.n_views - 1) / (self.n_views_use - 1))
            for i in range(self.n_views_use)
        ] if self.n_views_use > 1 else [self.n_views // 2]

        use_material_context = data_cfg.get("use_material_context", False)
        if use_material_context and "voxel_name_data" in df.columns:
            top_k = data_cfg.get("top_k_materials", 5)
            self.texts = [
                build_text_with_materials(df.iloc[i], top_k_materials=top_k)
                for i in range(len(df))
            ]
        else:
            self.texts = [build_text(df.iloc[i]) for i in range(len(df))]

        self.voxel_name_data = df["voxel_name_data"].tolist()
        self.categories = df["subtitle"].fillna("Unknown").tolist()

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        text = self.texts[idx]

        # load all n_views_use views → (N, 3, H, W)
        views = []
        for vi in self._view_indices:
            img_path = resolve_image_path(row, self.renders_base, vi, self.n_views)
            try:
                img = Image.open(img_path).convert("RGB") if img_path else self.fallback_img
            except Exception:
                img = self.fallback_img
            pv = self.processor(images=img, return_tensors="pt")["pixel_values"].squeeze(0)
            views.append(pv)
        pixel_values = torch.stack(views, dim=0)  # (N_views, 3, H, W)

        voxel = remap_voxel_from_names(
            self.voxel_name_data[idx], self.block_mapping, crop_bbox=self.crop_bbox
        )
        if self.aug:
            voxel = augment_voxel(voxel, self.aug_apply_prob, self.aug_dropout_prob)

        return text, pixel_values, voxel, self.categories[idx]


def _collate_trimodal(batch):
    texts, images, voxels, categories = zip(*batch)
    return list(texts), torch.stack(images), torch.stack(voxels), list(categories)


def create_trimodal_dataloaders(cfg: dict):
    """Load trimodal parquet, split train/val/test, return DataLoaders + metadata.

    Returns:
        train_loader, val_loader, test_loader, block_mapping, num_block_types, processor
    """
    data_cfg = cfg["data"]
    path = data_cfg["parquet_path"]
    clip_name = cfg["model"].get("clip_model", "openai/clip-vit-base-patch16")

    import pyarrow.parquet as pq
    available = set(pq.ParquetFile(path).schema_arrow.names)
    needed = {"subtitle", "title", "description", "tags", "voxel_name_data"}
    n_views = data_cfg.get("n_views", 12)
    for i in range(n_views):
        col = f"view_{i:02d}_path"
        if col in available:
            needed.add(col)
    if "render_folder" in available:
        needed.add("render_folder")
    if data_cfg.get("use_material_context", False) and "voxel_name_data" in available:
        needed.add("voxel_name_data")

    load_cols = sorted(needed & available)
    df = pd.read_parquet(path, columns=load_cols)
    print(f"[Trimodal] Loaded {len(df)} samples | columns: {load_cols}")

    max_types = data_cfg["max_block_types"]
    block_mapping = build_block_name_mapping(df["voxel_name_data"], max_types=max_types)
    num_block_types = max_types
    print(f"[Trimodal] Block vocab: {max_types} types")

    processor = CLIPProcessor.from_pretrained(clip_name)

    val_split = data_cfg.get("val_split", 0.1)
    test_split = data_cfg.get("test_split", 0.1)
    seed = data_cfg.get("seed", 42)

    labels = df["subtitle"].fillna("Unknown").tolist()
    idx_all = list(range(len(df)))
    idx_train, idx_temp, _, labels_temp = train_test_split(
        idx_all, labels,
        test_size=val_split + test_split,
        stratify=labels,
        random_state=seed,
    )
    val_frac = val_split / (val_split + test_split)
    idx_val, idx_test = train_test_split(
        idx_temp, test_size=1 - val_frac,
        stratify=labels_temp,
        random_state=seed,
    )
    print(f"[Trimodal] Split — train={len(idx_train)} val={len(idx_val)} test={len(idx_test)}")

    num_workers = cfg["training"].get("num_workers", 2)
    batch_size = cfg["training"]["batch_size"]

    def make_loader(indices, split):
        ds = TriModalDataset(df.iloc[indices], block_mapping, processor, cfg, split=split)
        return DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=(split == "train"),
            num_workers=num_workers,
            collate_fn=_collate_trimodal,
            pin_memory=torch.cuda.is_available(),
            drop_last=(split == "train"),
        )

    return (
        make_loader(idx_train, "train"),
        make_loader(idx_val, "val"),
        make_loader(idx_test, "test"),
        block_mapping,
        num_block_types,
        processor,
    )
