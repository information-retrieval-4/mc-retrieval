"""
WEMIR CBIR Demo
===============
This builds an index and allows for querying of images.

Usage:
    # Build index from image directory
    python demo.py build --data ./corel --output index.pkl

    # Query with an image
    python demo.py query --index index.pkl --image path/to/query.jpg --top_k 10

    # Evaluate precision/recall on random samples
    python demo.py evaluate --index index.pkl --data ./corel --samples 20
"""

import argparse
import sys
import random
from pathlib import Path

import cv2
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pywt

from wemir import (
    WEMIRIndex,
    median_filter,
    kmeans_cluster,
    select_largest_cluster,
    histogram_equalization,
    dwt_ll,
    svd_reduce,
    extract_block_features,
    hungarian_block,
)


def cmd_build(args):
    """Build the WEMIR index from a directory of images."""
    data_dir = Path(args.data)
    if not data_dir.exists():
        print(f"Error: data directory '{data_dir}' not found")
        sys.exit(1)

    index = WEMIRIndex(svd_rank=args.svd_rank)
    index.build(data_dir)
    index.save(args.output)


def cmd_query(args):
    """Query the index with an image and display results."""
    index = WEMIRIndex.load(args.index)

    results = index.query(
        args.image,
        top_k=args.top_k,
        metric=args.metric,
    )

    query_label = Path(args.image).parent.name
    print(f"\nQuery: {args.image} (category: {query_label})")
    print(f"Metric: {args.metric}")
    print(f"{'Rank':<6} {'Distance':<14} {'Category':<20} {'File'}")
    print("-" * 80)

    for rank, (path, dist, label) in enumerate(results, 1):
        match = "✓" if label == query_label else "✗"
        print(f"{rank:<6} {dist:<14.4f} {label:<20} {Path(path).name} {match}")

    # Show results visually
    if not args.no_display:
        _display_results(args.image, results, args.output_image)


def cmd_evaluate(args):
    """Evaluate precision/recall on random query samples."""
    index = WEMIRIndex.load(args.index)

    # Pick random images from the index as queries
    all_paths = list(index.features.keys())
    n_samples = min(args.samples, len(all_paths))
    sample_paths = random.sample(all_paths, n_samples)

    precisions = []
    recalls = []
    per_category = {}

    for qpath in sample_paths:
        result = index.evaluate(qpath, top_k=args.top_k, metric=args.metric)
        precisions.append(result["precision"])
        recalls.append(result["recall"])

        cat = result["query_label"]
        if cat not in per_category:
            per_category[cat] = {"precisions": [], "recalls": []}
        per_category[cat]["precisions"].append(result["precision"])
        per_category[cat]["recalls"].append(result["recall"])

    # Print per-category results
    print(f"\n{'Category':<20} {'Avg Precision':<16} {'Avg Recall':<16} {'Samples'}")
    print("-" * 68)

    for cat in sorted(per_category.keys()):
        data = per_category[cat]
        avg_p = np.mean(data["precisions"])
        avg_r = np.mean(data["recalls"])
        n = len(data["precisions"])
        print(f"{cat:<20} {avg_p:<16.4f} {avg_r:<16.4f} {n}")

    print("-" * 68)
    print(
        f"{'AVERAGE':<20} {np.mean(precisions):<16.4f} {np.mean(recalls):<16.4f} {n_samples}"
    )


def cmd_visual(args):
    """Visualize each step of the WEMIR pipeline with explanations."""
    image = cv2.imread(args.image)
    if image is None:
        print(f"Error: can't read image '{args.image}'")
        sys.exit(1)

    print(f"Running WEMIR visual explainer on: {args.image}")
    print(f"SVD rank: {args.svd_rank or 'auto (90% energy)'}")

    # use a non-interactive backend if saving to file
    if args.output:
        matplotlib.use("Agg")

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # ---------- step 1: median filter ----------
    filtered = median_filter(image)
    filtered_rgb = cv2.cvtColor(filtered, cv2.COLOR_BGR2RGB)

    # ---------- step 2: k-means clustering ----------
    labels, centers = kmeans_cluster(filtered, k=3)
    label_map = labels.reshape(image.shape[:2])
    # create a color-coded cluster visualization
    cluster_colors = np.array([[31, 119, 180], [255, 127, 14], [44, 160, 44]], dtype=np.uint8)
    cluster_vis = cluster_colors[label_map]

    # ---------- step 3: select largest cluster ----------
    cluster_img, mask = select_largest_cluster(filtered, labels)
    cluster_img_rgb = cv2.cvtColor(cluster_img, cv2.COLOR_BGR2RGB)
    # make a nice mask overlay
    mask_overlay = image_rgb.copy()
    mask_overlay[~mask] = (mask_overlay[~mask] * 0.25).astype(np.uint8)

    # ---------- step 4: histogram equalization ----------
    enhanced = histogram_equalization(cluster_img)
    enhanced_rgb = cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB)

    # ---------- step 5: DWT → LL subband ----------
    ll = dwt_ll(enhanced)

    # ---------- step 6: SVD reduction ----------
    reduced = svd_reduce(ll, rank=args.svd_rank)
    # compute how many singular values were kept
    U, S, Vt = np.linalg.svd(ll, full_matrices=False)
    total_energy = np.sum(S**2)
    cumulative = np.cumsum(S**2) / total_energy
    if args.svd_rank is None:
        rank_used = np.searchsorted(cumulative, 0.9) + 1
        rank_used = max(rank_used, 5)
    else:
        rank_used = args.svd_rank
    rank_used = min(rank_used, len(S))

    # ---------- step 7: block Hungarian assignment ----------
    features = extract_block_features(reduced)

    # visualize a sample 5x5 block + its assignment
    h, w = reduced.shape
    pad_h = (5 - h % 5) % 5
    pad_w = (5 - w % 5) % 5
    padded = np.pad(reduced, ((0, pad_h), (0, pad_w)), mode="constant") if pad_h or pad_w else reduced.copy()
    sample_block = padded[0:5, 0:5]
    from scipy.optimize import linear_sum_assignment
    cost = sample_block.copy()
    if cost.min() < 0:
        cost = cost - cost.min()
    row_ind, col_ind = linear_sum_assignment(cost)
    assigned_vals = sample_block[row_ind, col_ind]

    # ============ build the figure ============
    fig = plt.figure(figsize=(20, 24), facecolor="#0d1117")
    fig.suptitle(
        "WEMIR Pipeline — Step-by-Step Visualization",
        fontsize=22, fontweight="bold", color="#58a6ff",
        y=0.98,
    )

    gs = gridspec.GridSpec(5, 4, hspace=0.45, wspace=0.3,
                           top=0.95, bottom=0.02, left=0.05, right=0.95)

    text_props = dict(fontsize=9, color="#c9d1d9", family="monospace",
                      verticalalignment="top")
    title_props = dict(fontsize=12, fontweight="bold", color="#f0f6fc", pad=8)
    ax_style = {"facecolor": "#161b22"}

    def style_ax(ax, title):
        ax.set_title(title, **title_props)
        ax.set_facecolor("#161b22")
        ax.tick_params(colors="#484f58", labelsize=7)
        for spine in ax.spines.values():
            spine.set_color("#30363d")

    # --- row 0: original + median filter ---
    ax0 = fig.add_subplot(gs[0, 0])
    ax0.imshow(image_rgb)
    style_ax(ax0, "① Original Image")
    ax0.axis("off")

    ax1 = fig.add_subplot(gs[0, 1])
    ax1.imshow(filtered_rgb)
    style_ax(ax1, "② Median Filter (3×3)")
    ax1.axis("off")

    ax_t0 = fig.add_subplot(gs[0, 2:])
    style_ax(ax_t0, "Steps 1–2: Noise Removal")
    ax_t0.axis("off")
    ax_t0.text(0.05, 0.85,
        "Median filter removes salt-and-pepper noise while\n"
        "preserving edges. The 3×3 kernel replaces each pixel\n"
        "with the median of its neighborhood.\n\n"
        "This is the first stage of the preprocessing pipeline\n"
        "described in Tamilkodi & Nesakumari (2021), Section 2.",
        transform=ax_t0.transAxes, **text_props)

    # --- row 1: k-means + cluster selection ---
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.imshow(cluster_vis)
    style_ax(ax2, "③ K-Means (k=3)")
    ax2.axis("off")

    ax3 = fig.add_subplot(gs[1, 1])
    ax3.imshow(mask_overlay)
    style_ax(ax3, "④ Largest Cluster")
    ax3.axis("off")

    ax_t1 = fig.add_subplot(gs[1, 2:])
    style_ax(ax_t1, "Steps 3–4: Clustering & Selection")
    ax_t1.axis("off")
    counts = np.bincount(labels, minlength=3)
    pcts = counts / counts.sum() * 100
    cluster_info = "\n".join(f"  Cluster {i}: {counts[i]:>6} px ({pcts[i]:.1f}%)" for i in range(3))
    ax_t1.text(0.05, 0.85,
        f"K-Means groups all pixels into k=3 clusters based\n"
        f"on RGB Euclidean distance. The largest cluster is\n"
        f"selected as the dominant region of interest.\n\n"
        f"Cluster sizes:\n{cluster_info}\n\n"
        f"Selected: Cluster {np.argmax(counts)} (largest)",
        transform=ax_t1.transAxes, **text_props)

    # --- row 2: histogram eq + DWT ---
    ax4 = fig.add_subplot(gs[2, 0])
    ax4.imshow(enhanced_rgb)
    style_ax(ax4, "⑤ Histogram Equalization")
    ax4.axis("off")

    ax5 = fig.add_subplot(gs[2, 1])
    ax5.imshow(ll, cmap="gray")
    style_ax(ax5, "⑥ DWT LL Subband")
    ax5.axis("off")

    ax_t2 = fig.add_subplot(gs[2, 2:])
    style_ax(ax_t2, "Steps 5–6: Enhancement & Transform")
    ax_t2.axis("off")
    ax_t2.text(0.05, 0.85,
        f"Histogram equalization (confined mean computation)\n"
        f"redistributes intensity values for better contrast.\n\n"
        f"1-level Haar DWT decomposes the image into frequency\n"
        f"subbands. The LL (low-low) subband captures the\n"
        f"approximation at half resolution.\n\n"
        f"LL subband shape: {ll.shape[0]}×{ll.shape[1]}",
        transform=ax_t2.transAxes, **text_props)

    # --- row 3: SVD + singular value plot ---
    ax6 = fig.add_subplot(gs[3, 0])
    ax6.imshow(reduced, cmap="gray")
    style_ax(ax6, f"⑦ SVD Reduced (rank={rank_used})")
    ax6.axis("off")

    ax7 = fig.add_subplot(gs[3, 1])
    n_show = min(50, len(S))
    colors = ["#58a6ff" if i < rank_used else "#484f58" for i in range(n_show)]
    ax7.bar(range(n_show), S[:n_show], color=colors, width=0.8)
    ax7.axvline(x=rank_used - 0.5, color="#f85149", linestyle="--", linewidth=1.5, alpha=0.8)
    style_ax(ax7, "Singular Values")
    ax7.set_xlabel("Index", fontsize=8, color="#8b949e")
    ax7.set_ylabel("σ", fontsize=10, color="#8b949e")

    ax_t3 = fig.add_subplot(gs[3, 2:])
    style_ax(ax_t3, "Step 7: SVD Reduction")
    ax_t3.axis("off")
    ax_t3.text(0.05, 0.85,
        f"SVD factorizes the LL subband as I = UΣVᵀ, then\n"
        f"reconstructs using only the top-r singular values.\n"
        f"This removes noise and compresses the representation.\n\n"
        f"Total singular values: {len(S)}\n"
        f"Kept: {rank_used} (blue bars)\n"
        f"Energy retained: {cumulative[rank_used - 1] * 100:.1f}%\n"
        f"Reduced matrix: {reduced.shape[0]}×{reduced.shape[1]}",
        transform=ax_t3.transAxes, **text_props)

    # --- row 4: Hungarian block demo + feature vector ---
    ax8 = fig.add_subplot(gs[4, 0])
    ax8.imshow(sample_block, cmap="viridis", interpolation="nearest")
    # annotate each cell with its value
    for i in range(5):
        for j in range(5):
            val = sample_block[i, j]
            color = "white" if val < (sample_block.max() + sample_block.min()) / 2 else "black"
            ax8.text(j, i, f"{val:.0f}", ha="center", va="center",
                     fontsize=8, color=color, fontweight="bold")
    # highlight assigned cells
    for r, c in zip(row_ind, col_ind):
        rect = plt.Rectangle((c - 0.5, r - 0.5), 1, 1, linewidth=2.5,
                              edgecolor="#f85149", facecolor="none")
        ax8.add_patch(rect)
    style_ax(ax8, "⑧ Hungarian (sample 5×5)")
    ax8.set_xticks(range(5))
    ax8.set_yticks(range(5))

    ax9 = fig.add_subplot(gs[4, 1])
    feat_show = features[:100]  # show first 100 values
    ax9.plot(feat_show, color="#58a6ff", linewidth=0.8, alpha=0.9)
    ax9.fill_between(range(len(feat_show)), feat_show, alpha=0.15, color="#58a6ff")
    style_ax(ax9, "⑨ Feature Vector")
    ax9.set_xlabel("Index", fontsize=8, color="#8b949e")
    ax9.set_ylabel("Value", fontsize=8, color="#8b949e")

    ax_t4 = fig.add_subplot(gs[4, 2:])
    style_ax(ax_t4, "Steps 8–9: Hungarian Assignment → Features")
    ax_t4.axis("off")
    n_blocks_h = padded.shape[0] // 5
    n_blocks_w = padded.shape[1] // 5
    ax_t4.text(0.05, 0.85,
        f"The reduced matrix is split into non-overlapping\n"
        f"5×5 blocks. Each block is treated as a cost matrix\n"
        f"and solved via the Hungarian algorithm to find the\n"
        f"minimum-cost assignment (red boxes above).\n\n"
        f"Each block yields 5 intensity values → feature.\n\n"
        f"Grid: {n_blocks_h}×{n_blocks_w} = {n_blocks_h * n_blocks_w} blocks\n"
        f"Feature vector length: {len(features)}\n"
        f"Sample assigned values: [{', '.join(f'{v:.1f}' for v in assigned_vals)}]",
        transform=ax_t4.transAxes, **text_props)

    if args.output:
        fig.savefig(args.output, dpi=150, facecolor=fig.get_facecolor(),
                    bbox_inches="tight")
        print(f"Saved visualization to {args.output}")
        plt.close(fig)
    else:
        plt.show()


def _display_results(query_path, results, output_path=None):
    """Create a visual grid of query + results and display/save it."""
    query_img = cv2.imread(str(query_path))
    if query_img is None:
        return

    # Resize all to a uniform size for the grid
    thumb_size = (150, 150)
    query_thumb = cv2.resize(query_img, thumb_size)

    # Add green border to query
    cv2.rectangle(query_thumb, (0, 0), (149, 149), (0, 200, 0), 3)

    result_thumbs = []
    query_label = Path(query_path).parent.name

    for path, dist, label in results:
        img = cv2.imread(path)
        if img is None:
            continue
        thumb = cv2.resize(img, thumb_size)
        # Green border if same category, red if different
        color = (0, 200, 0) if label == query_label else (0, 0, 200)
        cv2.rectangle(thumb, (0, 0), (149, 149), color, 2)
        result_thumbs.append(thumb)

    if not result_thumbs:
        return

    # Build grid: query on top, results below in rows of 5
    cols = 5
    rows_needed = (len(result_thumbs) + cols - 1) // cols

    # Query row
    query_row = np.zeros((thumb_size[1] + 30, thumb_size[0] * cols, 3), dtype=np.uint8)
    query_row[30 : 30 + thumb_size[1], 0 : thumb_size[0]] = query_thumb
    cv2.putText(
        query_row, "QUERY", (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 0), 2
    )

    # Result rows
    result_grid = np.zeros(
        (rows_needed * (thumb_size[1] + 30), thumb_size[0] * cols, 3), dtype=np.uint8
    )
    for idx, thumb in enumerate(result_thumbs):
        r = idx // cols
        c = idx % cols
        y = r * (thumb_size[1] + 30) + 30
        x = c * thumb_size[0]
        result_grid[y : y + thumb_size[1], x : x + thumb_size[0]] = thumb

        # Rank label
        rank_text = f"#{idx + 1}"
        cv2.putText(
            result_grid,
            rank_text,
            (x + 5, y - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
        )

    # Stack vertically
    canvas = np.vstack([query_row, result_grid])

    if output_path:
        cv2.imwrite(str(output_path), canvas)
        print(f"\nResults saved to {output_path}")
    else:
        cv2.imshow("WEMIR Results", canvas)
        print("\nPress any key to close the results window...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(
        description="WEMIR - Weighted Edge Matching Information Retrieval"
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Build
    build_parser = subparsers.add_parser("build", help="Build feature index")
    build_parser.add_argument("--data", required=True, help="Path to image directory")
    build_parser.add_argument("--output", default="index.pkl", help="Output index file")
    build_parser.add_argument(
        "--svd_rank", type=int, default=None, help="SVD rank (None = auto)"
    )

    # Query
    query_parser = subparsers.add_parser("query", help="Query with an image")
    query_parser.add_argument("--index", required=True, help="Path to index file")
    query_parser.add_argument("--image", required=True, help="Query image path")
    query_parser.add_argument("--top_k", type=int, default=10, help="Number of results")
    query_parser.add_argument(
        "--metric", default="euclidean", choices=["euclidean", "manhattan"]
    )
    query_parser.add_argument(
        "--no_display", action="store_true", help="Skip visual display"
    )
    query_parser.add_argument(
        "--output_image",
        default=None,
        help="Save result grid to file instead of displaying",
    )

    # Evaluate
    eval_parser = subparsers.add_parser("evaluate", help="Evaluate precision/recall")
    eval_parser.add_argument("--index", required=True, help="Path to index file")
    eval_parser.add_argument(
        "--data", default=None, help="Image directory (for finding categories)"
    )
    eval_parser.add_argument(
        "--samples", type=int, default=20, help="Number of random query samples"
    )
    eval_parser.add_argument("--top_k", type=int, default=10, help="Number of results")
    eval_parser.add_argument(
        "--metric", default="euclidean", choices=["euclidean", "manhattan"]
    )

    # Visual
    visual_parser = subparsers.add_parser(
        "visual", help="Visualize the WEMIR pipeline step by step"
    )
    visual_parser.add_argument("--image", required=True, help="Image to visualize")
    visual_parser.add_argument(
        "--svd_rank", type=int, default=None, help="SVD rank (None = auto)"
    )
    visual_parser.add_argument(
        "--output", default=None, help="Save figure to file instead of displaying"
    )

    args = parser.parse_args()

    if args.command == "build":
        cmd_build(args)
    elif args.command == "query":
        cmd_query(args)
    elif args.command == "evaluate":
        cmd_evaluate(args)
    elif args.command == "visual":
        cmd_visual(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
