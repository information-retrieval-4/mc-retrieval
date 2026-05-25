"""Visualize the shared text-voxel embedding space using UMAP."""

import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import umap
from tqdm import tqdm

from dataset import create_dataloaders
from model import DualEncoder
from utils import load_config, set_seed, get_device, load_checkpoint


@torch.no_grad()
def extract_embeddings(model, loader, device):
    """Extract text and voxel embeddings + categories from a data loader."""
    model.eval()
    text_embs, voxel_embs, categories = [], [], []

    for texts, voxels, cats in tqdm(loader, desc="Extracting embeddings", leave=False):
        voxels = voxels.to(device)
        t_emb, v_emb = model(texts, voxels)
        text_embs.append(t_emb.cpu())
        voxel_embs.append(v_emb.cpu())
        categories.extend(cats)

    return (torch.cat(text_embs, 0).numpy(),
            torch.cat(voxel_embs, 0).numpy(),
            np.array(categories))


def plot_embedding_space(text_embs, voxel_embs, categories, save_path="embedding_space.png"):
    """Create UMAP visualization of the shared embedding space."""
    # combine all embeddings for joint UMAP
    all_embs = np.concatenate([text_embs, voxel_embs], axis=0)
    N = len(text_embs)
    modalities = np.array(["text"] * N + ["voxel"] * N)
    all_cats = np.concatenate([categories, categories])

    # UMAP projection
    print("Running UMAP...")
    reducer = umap.UMAP(
        n_neighbors=30,
        min_dist=0.3,
        metric="cosine",
        random_state=42,
    )
    coords = reducer.fit_transform(all_embs)

    # pick top categories by frequency for clean visualization
    unique_cats, counts = np.unique(categories, return_counts=True)
    top_k = 8
    top_cats = unique_cats[np.argsort(-counts)][:top_k]

    # color palette
    cmap = plt.cm.get_cmap("tab10", top_k + 1)
    cat_colors = {cat: cmap(i) for i, cat in enumerate(top_cats)}
    cat_colors["Other"] = (0.7, 0.7, 0.7, 0.4)

    # assign colors
    colors = []
    for cat in all_cats:
        colors.append(cat_colors.get(cat, cat_colors["Other"]))

    # shorten category names for legend
    short_names = {
        "Land Structure Map": "Land Structure",
        "3D Art Map": "3D Art",
        "Redstone Device Map": "Redstone Device",
        "Other Map": "Other Map",
        "Air Structure Map": "Air Structure",
        "Complex Map": "Complex",
        "Pixel Art Map": "Pixel Art",
        "Water Structure Map": "Water Structure",
        "Environment / Landscaping Map": "Environment",
        "Piston Map": "Piston",
    }

    # --- plot ---
    fig, axes = plt.subplots(1, 3, figsize=(24, 7))

    # Plot 1: all points, colored by category
    ax = axes[0]
    ax.set_title("Embedding Space — by Category", fontsize=14, fontweight="bold")
    for i, cat in enumerate(top_cats):
        mask = all_cats == cat
        ax.scatter(coords[mask, 0], coords[mask, 1],
                   c=[cat_colors[cat]], s=8, alpha=0.5,
                   label=short_names.get(cat, cat))
    # plot "other" categories
    other_mask = ~np.isin(all_cats, top_cats)
    if other_mask.any():
        ax.scatter(coords[other_mask, 0], coords[other_mask, 1],
                   c=[(0.7, 0.7, 0.7, 0.3)], s=4, alpha=0.2, label="Other")
    ax.legend(fontsize=8, markerscale=3, loc="best")
    ax.set_xticks([])
    ax.set_yticks([])

    # Plot 2: colored by modality (text vs voxel)
    ax = axes[1]
    ax.set_title("Embedding Space — by Modality", fontsize=14, fontweight="bold")
    text_mask = modalities == "text"
    voxel_mask = modalities == "voxel"
    ax.scatter(coords[voxel_mask, 0], coords[voxel_mask, 1],
               c="#2196F3", s=8, alpha=0.4, label="Voxel")
    ax.scatter(coords[text_mask, 0], coords[text_mask, 1],
               c="#FF5722", s=8, alpha=0.4, marker="x", label="Text")
    ax.legend(fontsize=10, markerscale=3)
    ax.set_xticks([])
    ax.set_yticks([])

    # Plot 3: paired connections for a subset
    ax = axes[2]
    ax.set_title("Text-Voxel Pairs (subset)", fontsize=14, fontweight="bold")
    # show 80 random pairs with lines connecting them
    np.random.seed(42)
    sample_idx = np.random.choice(N, min(80, N), replace=False)

    # plot all points faded
    ax.scatter(coords[:N, 0], coords[:N, 1],
               c="#FF5722", s=6, alpha=0.1, marker="x")
    ax.scatter(coords[N:, 0], coords[N:, 1],
               c="#2196F3", s=6, alpha=0.1)

    # draw lines and highlight sampled pairs
    for idx in sample_idx:
        cat = categories[idx]
        color = cat_colors.get(cat, (0.5, 0.5, 0.5, 0.5))
        # text point
        tx, ty = coords[idx, 0], coords[idx, 1]
        # voxel point
        vx, vy = coords[N + idx, 0], coords[N + idx, 1]
        ax.plot([tx, vx], [ty, vy], c=color, alpha=0.3, linewidth=0.5)
        ax.scatter(tx, ty, c=[color], s=20, marker="x", zorder=5)
        ax.scatter(vx, vy, c=[color], s=20, marker="o", zorder=5)

    legend_elements = [
        Line2D([0], [0], marker="x", color="w", markeredgecolor="#FF5722",
               markersize=8, label="Text"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#2196F3",
               markersize=8, label="Voxel"),
        Line2D([0], [0], color="gray", alpha=0.5, label="Paired"),
    ]
    ax.legend(handles=legend_elements, fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    print(f"Saved to {save_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Visualize embedding space")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best.pt")
    parser.add_argument("--output", type=str, default="embedding_space.png")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["data"]["seed"])
    device = get_device()

    _, _, test_loader, block_mapping, num_blocks = create_dataloaders(cfg)

    ckpt = load_checkpoint(args.checkpoint, device)
    model = DualEncoder(cfg, num_block_types=num_blocks).to(device)
    model.load_state_dict(ckpt["model_state"])

    text_embs, voxel_embs, categories = extract_embeddings(model, test_loader, device)
    print(f"Extracted {len(text_embs)} embeddings (dim={text_embs.shape[1]})")

    plot_embedding_space(text_embs, voxel_embs, categories, args.output)


if __name__ == "__main__":
    main()
