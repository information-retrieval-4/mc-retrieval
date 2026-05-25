"""Visualize MVM reconstruction with 3D voxel rendering."""

import argparse
import colorsys
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
import pandas as pd

from pretrain import MaskedVoxelModel, VoxelOnlyDataset, create_mask
from dataset import build_block_mapping
from utils import load_config, set_seed, get_device


def make_block_palette(n_blocks=256):
    """Generate distinct colors for each block ID."""
    colors = {0: (0, 0, 0, 0)}  # air = transparent
    for i in range(1, n_blocks):
        hue = (i * 0.618033988749895) % 1.0
        sat = 0.45 + 0.35 * ((i * 7) % 10) / 10
        val = 0.55 + 0.35 * ((i * 3) % 10) / 10
        r, g, b = colorsys.hsv_to_rgb(hue, sat, val)
        colors[i] = (r, g, b, 1.0)
    return colors


def vol_to_colors(vol, palette):
    """Convert block ID volume to RGBA color volume."""
    shape = vol.shape
    rgba = np.zeros((*shape, 4))
    for bid, color in palette.items():
        mask = vol == bid
        if mask.any():
            rgba[mask] = color
    return rgba


def render_voxel_3d(ax, vol, palette, title="", elev=25, azim=135):
    """Render a voxel grid on a 3D axis."""
    filled = vol != 0
    colors = vol_to_colors(vol, palette)

    # add slight edge darkening for depth
    edge_color = np.zeros((*vol.shape, 4))
    edge_color[filled] = [0, 0, 0, 0.08]

    ax.voxels(filled, facecolors=colors, edgecolors=edge_color, linewidth=0.1)
    ax.set_title(title, fontsize=10, fontweight="bold", pad=2)
    ax.view_init(elev=elev, azim=azim)
    ax.set_xlim(0, vol.shape[0])
    ax.set_ylim(0, vol.shape[1])
    ax.set_zlim(0, vol.shape[2])
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
    ax.set_box_aspect([1, 1, 1])
    # lighten the pane backgrounds
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor((0.9, 0.9, 0.9, 0.5))
    ax.yaxis.pane.set_edgecolor((0.9, 0.9, 0.9, 0.5))
    ax.zaxis.pane.set_edgecolor((0.9, 0.9, 0.9, 0.5))
    ax.grid(False)


@torch.no_grad()
def reconstruct(model, voxel, mask_ratio, device):
    """Run MVM reconstruction at a given mask ratio."""
    voxel = voxel.unsqueeze(0).to(device)
    mask = create_mask(voxel, mask_ratio=mask_ratio)
    masked = voxel.clone()
    masked[mask] = model.mask_token_id

    x = model.block_embedding(masked)
    x = x.permute(0, 4, 1, 2, 3).contiguous()
    e1 = model.enc1(x)
    e2 = model.enc2(model.pool1(e1))
    bn = model.bottleneck(model.pool2(e2))
    d2 = model.up2(bn)
    d2 = model.dec2(torch.cat([d2, e2], dim=1))
    d1 = model.up1(d2)
    d1 = model.dec1(torch.cat([d1, e1], dim=1))
    logits = model.pred_head(d1)
    recon = logits.argmax(dim=1)[0].cpu().numpy()

    m = mask[0].cpu().numpy()
    acc = (recon[m] == voxel[0].cpu().numpy()[m]).mean() if m.sum() > 0 else 1.0
    return recon, acc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/pretrained_voxel.pt")
    parser.add_argument("--output", type=str, default="mvm_reconstruction_3d.png")
    parser.add_argument("--sample-idx", type=int, default=None,
                        help="Specific sample index, or auto-picks a dense one")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["data"]["seed"])
    device = get_device()

    df = pd.read_parquet(cfg["data"]["parquet_path"])
    block_mapping = build_block_mapping(
        df["voxel_data"], max_types=cfg["data"]["max_block_types"]
    )
    test_df = df.iloc[int(len(df) * 0.9):]
    dataset = VoxelOnlyDataset(test_df, block_mapping)

    # pick a sample with good density
    if args.sample_idx is not None:
        sample = dataset[args.sample_idx]
    else:
        best_idx, best_density = 0, 0
        for i in range(min(200, len(dataset))):
            v = dataset[i]
            d = (v != 0).float().mean().item()
            if d > best_density:
                best_density = d
                best_idx = i
        sample = dataset[best_idx]
        print(f"Auto-picked sample {best_idx} (density={best_density:.1%})")

    original = sample.numpy()

    # load model
    model_cfg = cfg["model"]
    model = MaskedVoxelModel(
        num_block_types=cfg["data"]["max_block_types"],
        block_embed_dim=model_cfg["block_embed_dim"],
        channels=model_cfg["voxel_channels"],
        dropout=0.0,
        mask_ratio=0.2,
    ).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"Loaded checkpoint (epoch {ckpt['epoch']})")

    palette = make_block_palette(256)
    mask_ratios = [0.2, 0.5, 0.7, 0.9]

    # layout: 1 row, original + n reconstructions
    n_cols = 1 + len(mask_ratios)
    fig = plt.figure(figsize=(5 * n_cols, 5.5), facecolor="white")

    # original
    ax = fig.add_subplot(1, n_cols, 1, projection="3d")
    render_voxel_3d(ax, original, palette, title="Original")

    # reconstructions at each ratio
    for j, ratio in enumerate(mask_ratios):
        recon, acc = reconstruct(model, sample, ratio, device)
        ax = fig.add_subplot(1, n_cols, j + 2, projection="3d")
        render_voxel_3d(ax, recon, palette,
                        title=f"Mask {ratio:.0%}  (acc={acc:.1%})")

    fig.suptitle("Masked Voxel Modeling — Reconstruction Quality",
                 fontsize=16, fontweight="bold", y=0.98)
    plt.tight_layout(pad=1)
    plt.savefig(args.output, dpi=200, bbox_inches="tight", facecolor="white")
    print(f"Saved to {args.output}")
    plt.close()


if __name__ == "__main__":
    main()
