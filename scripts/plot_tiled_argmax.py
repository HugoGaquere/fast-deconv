#!/usr/bin/env python
"""Plot the tiled-argmax tile-size sweep from bench_tiled_argmax --csv output.

The benchmark times three argmaxes on the same data: the native full-image
cub::DeviceReduce::ArgMax (the production baseline the clean loop runs today), the
tiled full pass, and the incremental (dirty-footprint) pass, across a range of
tile sizes. The incremental pass is U-shaped in tile size:

  - small tiles -> many tiles -> the final combine rescans more cached slots, and
                   more tiles fall inside the footprint;
  - large tiles -> each dirty tile recomputes more clean pixels (wasted work).

The output is a 1x3 figure:
    (a) wall time per call vs tile size  (native ref line + tiled full + incremental)
    (b) speedup of incremental vs the native argmax, with the best tile marked
    (c) tile counts vs tile size         (total tiles + recomputed dirty tiles)

Usage (always via the project venv):
    .venv/bin/python scripts/plot_tiled_argmax.py out.csv [--outdir plots]
        [--yscale linear|log]
"""
import argparse
import csv
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FuncFormatter

INT_COLS = {
    "width", "height", "psf", "tile", "n_tiles_x", "n_tiles_y", "n_tiles",
    "dirty_tiles", "reps", "warmup",
}


def load(path):
    rows = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            rows.append({k: (int(v) if k in INT_COLS else float(v)) for k, v in row.items()})
    rows.sort(key=lambda r: r["tile"])
    return rows


def fmt_tile(n):
    """Compact tile-side tick label: 1024 -> '1k', 256 -> '256'."""
    return f"{n / 1024:g}k" if n >= 1024 else str(int(n))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--outdir", default="plots")
    ap.add_argument("--yscale", choices=("linear", "log"), default="log",
                    help="scale of the time axis in panel (a) (default log)")
    args = ap.parse_args()

    rows = load(args.csv)
    if len(rows) < 2:
        raise SystemExit("Need >=2 tile sizes in the CSV to plot a sweep.")

    tiles = [r["tile"] for r in rows]
    full = [r["full_mean_ms"] for r in rows]
    full_lo = [r["full_min_ms"] for r in rows]
    full_hi = [r["full_max_ms"] for r in rows]
    incr = [r["incr_mean_ms"] for r in rows]
    incr_lo = [r["incr_min_ms"] for r in rows]
    incr_hi = [r["incr_max_ms"] for r in rows]
    speedup = [r["speedup_incr"] for r in rows]  # incremental vs native cub argmax
    n_tiles = [r["n_tiles"] for r in rows]
    dirty = [r["dirty_tiles"] for r in rows]

    width, height, psf = rows[0]["width"], rows[0]["height"], rows[0]["psf"]
    native_ms = rows[0]["native_mean_ms"]  # constant across tiles -> reference line

    os.makedirs(args.outdir, exist_ok=True)
    tick_fmt = FuncFormatter(lambda x, _: fmt_tile(round(x)))

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), layout="constrained")

    # (a) wall time per call: native baseline (flat), tiled full, incremental.
    ax = axes[0]
    ax.axhline(native_ms, color="tab:blue", ls="--", lw=1.4,
               label=f"native cub argmax ({native_ms:.2f} ms)")
    ax.fill_between(tiles, full_lo, full_hi, color="tab:gray", alpha=0.15)
    ax.plot(tiles, full, color="tab:gray", marker="o", ms=4, label="tiled full")
    ax.fill_between(tiles, incr_lo, incr_hi, color="tab:orange", alpha=0.15)
    ax.plot(tiles, incr, color="tab:orange", marker="o", ms=4, label="incremental")
    ax.set_yscale(args.yscale)
    ax.set_ylabel("wall time / call (ms)")
    ax.set_title("(a) per-call time")
    ax.legend(fontsize=8)

    # (b) speedup of incremental vs the native argmax, best tile annotated. A
    # break-even line at 1x marks where incremental stops being worth it.
    ax = axes[1]
    ax.plot(tiles, speedup, color="tab:green", marker="o", ms=4)
    ax.axhline(1.0, color="0.5", ls=":", lw=1.0)
    best = max(range(len(tiles)), key=lambda i: speedup[i])
    ax.scatter([tiles[best]], [speedup[best]], color="tab:red", zorder=5)
    # Annotate to the lower-right of the peak so it never collides with the title
    # (the best point usually sits at the top of the axis); add top headroom too.
    ax.annotate(f"best: tile={tiles[best]}, {speedup[best]:.1f}x",
                xy=(tiles[best], speedup[best]), xytext=(8, -10),
                textcoords="offset points", ha="left", va="top", fontsize=9,
                fontweight="bold")
    ax.set_ylim(0, max(speedup) * 1.12)
    ax.set_ylabel("speedup (native argmax / incremental)")
    ax.set_title("(b) incremental speedup vs native")

    # (c) tile counts: total tiles (final-combine cost) vs recomputed dirty tiles.
    ax = axes[2]
    ax.plot(tiles, n_tiles, color="tab:purple", marker="o", ms=4, label="total tiles")
    ax.plot(tiles, dirty, color="tab:brown", marker="s", ms=4, label="dirty tiles recomputed")
    ax.set_yscale("log")
    ax.set_ylabel("count")
    ax.set_title("(c) tile counts")
    ax.legend(fontsize=9)

    for ax in axes:
        ax.set_xscale("log", base=2)
        ax.xaxis.set_major_locator(FixedLocator(tiles))
        ax.xaxis.set_major_formatter(tick_fmt)
        ax.set_xlabel("tile size (px, log2)")
        ax.tick_params(axis="x", labelrotation=45, labelsize=8)
        ax.grid(True, which="both", alpha=0.3)

    fig.suptitle(f"Tiled argmax: tile-size sweep  |  image {width}x{height}, footprint "
                 f"{psf}x{psf}  |  native cub ArgMax = {native_ms:.3f} ms",
                 fontsize=12, fontweight="bold")
    out = os.path.join(args.outdir, "tiled_argmax_tile_sweep.png")
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"[ok] {out}  ({len(tiles)} tile sizes)")


if __name__ == "__main__":
    main()
