"""Visualize MVM reconstruction at different mask ratios on sample voxels."""

import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib.gridspec as gridspec

from pretrain import MaskedVoxelModel, VoxelOnlyDataset, create_mask
from dataset import build_block_mapping
from utils import load_config, set_seed, get_device
import pandas as pd


def make_block_colormap(n_blocks=256):
    """Create a distinctive colormap for block types."""
    np.random.seed(0)
    # air = white, then random distinct colors
    colors = [(1, 1, 1, 1)]  # block 0 = air = white
    for i in range(1, n_blocks):
        hue = (i * 0.618033988749895) % 1.0  # golden ratio for spread
        sat = 0.5 + 0.5 * ((i * 7) % 10) / 10
        val = 0.6 + 0.4 * ((i * 3) % 10) / 10
        # HSV to RGB
        import colorsys
        r, g, b = colorsys.hsv_to_rgb(hue, sat, val)
        colors.append((r, g, b, 1))
    return ListedColormap(colors)


def render_topdown(vol):
    """Top-down view: color of highest non-air block at each (x, z)."""
    # vol shape: (32, 32, 32) = (X, Y, Z)
    # look down Y axis, find topmost non-air block
    result = np.zeros((vol.shape[0], vol.shape[2]), dtype=vol.dtype)
    for y in range(vol.shape[1] - 1, -1, -1):
        layer = vol[:, y, :]
        mask = (layer != 0) & (result == 0)
        result[mask] = layer[mask]
    return result


def render_side(vol):
    """Side view (X-Y plane): color of first non-air block along Z."""
    result = np.zeros((vol.shape[0], vol.shape[1]), dtype=vol.dtype)
    for z in range(vol.shape[2] - 1, -1, -1):
        layer = vol[:, :, z]
        mask = (layer != 0) & (result == 0)
        result[mask] = layer[mask]
    return result


@torch.no_grad()
def visualize_reconstruction(model, voxels_batch, device, save_path, cmap,
                              mask_ratios=[0.1, 0.3, 0.5, 0.7, 0.9],
                              n_samples=3):
    """Visualize original → masked → reconstructed for several samples and ratios."""
    model.eval()
    voxels_batch = voxels_batch.to(device)

    fig = plt.figure(figsize=(4 * len(mask_ratios), 4 * n_samples * 2), facecolor="white")

    # for each sample, show two rows: top-down and side view
    outer = gridspec.GridSpec(n_samples, 1, hspace=0.35, figure=fig)

    for s in range(n_samples):
        voxel = voxels_batch[s:s+1]  # (1, 32, 32, 32)
        original = voxel[0].cpu().numpy()

        inner = gridspec.GridSpecFromSubplotSpec(
            2, len(mask_ratios) + 1,
            subplot_spec=outer[s],
            wspace=0.05, hspace=0.08,
        )

        # original column
        ax = fig.add_subplot(inner[0, 0])
        ax.imshow(render_topdown(original), cmap=cmap, vmin=0, vmax=255,
                  interpolation="nearest")
        ax.set_title("Original", fontsize=9, fontweight="bold")
        ax.set_ylabel(f"Sample {s+1}\n(top)", fontsize=8)
        ax.set_xticks([]); ax.set_yticks([])

        ax = fig.add_subplot(inner[1, 0])
        ax.imshow(render_side(original).T, cmap=cmap, vmin=0, vmax=255,
                  interpolation="nearest", origin="lower")
        ax.set_ylabel("(side)", fontsize=8)
        ax.set_xticks([]); ax.set_yticks([])

        # each mask ratio
        for j, ratio in enumerate(mask_ratios):
            mask = create_mask(voxel, mask_ratio=ratio)
            masked = voxel.clone()
            masked[mask] = model.mask_token_id

            # reconstruct
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

            # blend: show reconstruction everywhere, but highlight errors
            display = recon.copy()

            # top-down
            ax = fig.add_subplot(inner[0, j + 1])
            ax.imshow(render_topdown(display), cmap=cmap, vmin=0, vmax=255,
                      interpolation="nearest")
            # compute accuracy for this sample
            m = mask[0].cpu().numpy()
            acc = (recon[m] == original[m]).mean() if m.sum() > 0 else 1.0
            ax.set_title(f"Mask {ratio:.0%}\nacc={acc:.1%}", fontsize=8)
            ax.set_xticks([]); ax.set_yticks([])

            # side view
            ax = fig.add_subplot(inner[1, j + 1])
            ax.imshow(render_side(display).T, cmap=cmap, vmin=0, vmax=255,
                      interpolation="nearest", origin="lower")
            ax.set_xticks([]); ax.set_yticks([])

    fig.suptitle("MVM Reconstruction at Various Mask Ratios",
                 fontsize=16, fontweight="bold", y=0.98)
    plt.savefig(save_path, dpi=200, bbox_inches="tight", facecolor="white")
    print(f"Saved to {save_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/pretrained_voxel.pt")
    parser.add_argument("--output", type=str, default="mvm_reconstruction.png")
    parser.add_argument("--samples", type=int, default=3)
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["data"]["seed"])
    device = get_device()

    df = pd.read_parquet(cfg["data"]["parquet_path"])
    block_mapping = build_block_mapping(
        df["voxel_data"], max_types=cfg["data"]["max_block_types"]
    )

    # use test split samples
    n = len(df)
    test_df = df.iloc[int(n * 0.9):]

    dataset = VoxelOnlyDataset(test_df, block_mapping)
    # pick samples with decent non-air content
    samples = []
    for i in range(len(dataset)):
        v = dataset[i]
        density = (v != 0).float().mean().item()
        if density > 0.02:
            samples.append(v)
        if len(samples) >= args.samples:
            break

    voxels = torch.stack(samples)

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
    print(f"Loaded checkpoint (epoch {ckpt['epoch']})")

    cmap = make_block_colormap(256)
    visualize_reconstruction(model, voxels, device, args.output, cmap,
                              n_samples=args.samples)


if __name__ == "__main__":
    main()
