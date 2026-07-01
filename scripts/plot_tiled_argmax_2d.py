#!/usr/bin/env python
"""Plot the tiled-argmax 2D sweep (tile size x footprint) from bench_tiled_argmax.

Run the benchmark with both --tiles and --psfs to get one CSV row per
(footprint, tile) pair. This script renders:

    (a) speedup heatmap (incremental vs the native cub::DeviceReduce::ArgMax
        baseline the clean loop runs today), best tile per footprint outlined;
    (b) incremental wall-time heatmap (log colour, lower = better);
    (c) optimal tile size vs footprint -- the headline: how the best tile grows
        with the dirtied PSF footprint.

The speedup is taken against `native` (a fixed per-image constant), so the best
tile per row is simply the one with the lowest incremental time -- it does not
suffer the artifact of dividing by the tiled full pass, which craters for tiny
tiles.

The optimum balances two opposing costs: the single-block final combine scales
with the tile *count* (~image_pixels / tile^2, favouring large tiles), while the
incremental recompute scales with the dirtied *area* (~(footprint + tile)^2,
favouring small tiles). Where they cross is the per-footprint optimum in (c).

Usage (always via the project venv):
    .venv/bin/python scripts/plot_tiled_argmax_2d.py out.csv [--outdir plots]
"""
import argparse
import csv
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

INT_COLS = {
    "width", "height", "psf", "tile", "n_tiles_x", "n_tiles_y", "n_tiles",
    "dirty_tiles", "reps", "warmup",
}


def load(path):
    rows = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            rows.append({k: (int(v) if k in INT_COLS else float(v)) for k, v in row.items()})
    return rows


def fmt_px(n):
    """Compact pixel label: 1024 -> '1k', 1700 -> '1.7k', 256 -> '256'."""
    return f"{n / 1024:g}k" if n >= 1024 else str(int(n))


def grid(rows, tiles, psfs, key):
    """Build a [psf, tile] array of `key`, NaN where a config is missing."""
    ti = {t: j for j, t in enumerate(tiles)}
    pi = {p: i for i, p in enumerate(psfs)}
    arr = np.full((len(psfs), len(tiles)), np.nan)
    for r in rows:
        arr[pi[r["psf"]], ti[r["tile"]]] = r[key]
    return arr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--outdir", default="plots")
    args = ap.parse_args()

    rows = load(args.csv)
    tiles = sorted({r["tile"] for r in rows})
    psfs = sorted({r["psf"] for r in rows})
    if len(tiles) < 2 or len(psfs) < 2:
        raise SystemExit("Need >=2 tile sizes AND >=2 footprints for a 2D sweep "
                         "(run bench_tiled_argmax with --tiles and --psfs).")

    # Speedup is incremental vs the native cub::DeviceReduce::ArgMax baseline (a
    # fixed per-image constant), so max speedup == min incremental time -- the true
    # optimum, free of the tiled-full-pass artifact that broke argmax(full/incr).
    speedup = grid(rows, tiles, psfs, "speedup_incr")
    incr = grid(rows, tiles, psfs, "incr_mean_ms")
    width, height = rows[0]["width"], rows[0]["height"]
    native_ms = rows[0]["native_mean_ms"]

    # Best tile per footprint (row), by max speedup-vs-native -- ignoring NaN.
    best_j = np.nanargmax(speedup, axis=1)
    best_tiles = [tiles[j] for j in best_j]

    os.makedirs(args.outdir, exist_ok=True)
    xt = np.arange(len(tiles))
    yt = np.arange(len(psfs))
    xlabels = [fmt_px(t) for t in tiles]
    ylabels = [fmt_px(p) for p in psfs]

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), layout="constrained")

    def heat(ax, arr, title, cmap, log, valfmt):
        norm = matplotlib.colors.LogNorm() if log else None
        im = ax.imshow(arr, aspect="auto", origin="lower", cmap=cmap, norm=norm)
        ax.set_xticks(xt, xlabels)
        ax.set_yticks(yt, ylabels)
        ax.set_xlabel("tile size (px)")
        ax.set_ylabel("footprint (px)")
        ax.set_title(title)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        for i in range(arr.shape[0]):
            for j in range(arr.shape[1]):
                if np.isnan(arr[i, j]):
                    continue
                ax.text(j, i, valfmt(arr[i, j]), ha="center", va="center", fontsize=7,
                        color="white" if _dark(im, arr[i, j]) else "black")
        return im

    # (a) speedup heatmap (incremental vs native argmax), best tile per footprint
    # outlined in red. Values <1 (red end) mean incremental is SLOWER than native.
    heat(axes[0], speedup, "(a) incremental speedup vs native argmax", "viridis", False,
         lambda v: f"{v:.0f}")
    for i, j in enumerate(best_j):
        axes[0].add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, ec="red", lw=2))

    # (b) incremental time heatmap (log colour; lower is better).
    heat(axes[1], incr, "(b) incremental time (ms)", "viridis_r", True,
         lambda v: f"{v:.3g}")

    # (c) optimal tile vs footprint.
    ax = axes[2]
    ax.plot(psfs, best_tiles, color="tab:red", marker="o", ms=6)
    for p, t in zip(psfs, best_tiles):
        ax.annotate(f"{t}", xy=(p, t), xytext=(0, 7), textcoords="offset points",
                    ha="center", fontsize=8, fontweight="bold")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log", base=2)
    ax.set_xticks(psfs, ylabels)
    ax.set_yticks(tiles, xlabels)
    ax.set_xlabel("footprint (px, log2)")
    ax.set_ylabel("optimal tile size (px, log2)")
    ax.set_title("(c) optimal tile vs footprint")
    ax.grid(True, which="both", alpha=0.3)

    fig.suptitle(f"Tiled argmax: tile x footprint sweep  |  image {width}x{height}  |  "
                 f"native cub ArgMax = {native_ms:.3f} ms",
                 fontsize=12, fontweight="bold")
    out = os.path.join(args.outdir, "tiled_argmax_2d_sweep.png")
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"[ok] {out}  ({len(psfs)} footprints x {len(tiles)} tiles)")


def _dark(im, v):
    """True when the colormapped cell is dark enough to need white text."""
    r, g, b, _ = im.cmap(im.norm(v))
    return 0.299 * r + 0.587 * g + 0.114 * b < 0.5


if __name__ == "__main__":
    main()
