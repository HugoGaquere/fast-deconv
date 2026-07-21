#!/usr/bin/env python
"""Amortized tiled-argmax analysis: effective speedup vs reseed interval N.

The incremental path needs a one-time full tiled pass to seed the per-tile cache,
then handles N dirty-footprint updates before the cache must be rebuilt (in DDMSC
a scale switch dirties the whole image, forcing a full recompute). So per argmax
call the tiled approach costs, amortized over a block of N iterations:

    amortized(tile, N) = full(tile) / N  +  incr(tile)

versus the native cub::DeviceReduce::ArgMax baseline, which pays its full cost on
every call. The effective speedup is therefore

    eff_speedup(tile, N) = native / (full(tile)/N + incr(tile))

which climbs from below 1x at N=1 (a single reseed already costs more than one
native call) toward native/incr as N -> inf. Because `full` and `incr` favour
different tiles, the optimal tile shifts with N: small N favours the tile with the
cheapest full pass, large N the tile with the cheapest incremental update.

This is pure post-processing of the bench CSV (native/full/incr per tile) -- no
extra timing is needed, the amortization is arithmetic.

The output is a 1x2 figure for one footprint (N on a log axis, speedup linear):
    (a) effective speedup vs N, one line per competitive tile, 1x break-even line;
    (b) best achievable speedup vs N (upper envelope over tiles), with the optimal
        tile shaded + labelled in each N region.

Usage (always via the project venv):
    .venv/bin/python scripts/plot_tiled_argmax_amortized.py out.csv
        [--psf P] [--ns 1,2,5,10,20,50,100,200,500,1000] [--outdir plots]
"""
import argparse
import csv
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--psf", type=int, default=None,
                    help="footprint to analyse (default: the only one, or the median)")
    ap.add_argument("--ns", default="1,2,5,10,20,50,100,200,500,1000",
                    help="comma-separated reseed intervals N for the (categorical) x-axis")
    ap.add_argument("--outdir", default="plots")
    args = ap.parse_args()

    rows = load(args.csv)
    psfs = sorted({r["psf"] for r in rows})
    if args.psf is not None:
        if args.psf not in psfs:
            raise SystemExit(f"psf={args.psf} not in CSV (have {psfs})")
        psf = args.psf
    elif len(psfs) == 1:
        psf = psfs[0]
    else:
        psf = psfs[len(psfs) // 2]
        print(f"[warn] multiple footprints {psfs}; using median psf={psf} (pass --psf to pick)")

    sel = sorted((r for r in rows if r["psf"] == psf), key=lambda r: r["tile"])
    if len(sel) < 2:
        raise SystemExit(f"Need >=2 tile sizes for psf={psf}.")

    tiles = np.array([r["tile"] for r in sel])
    full = np.array([r["full_mean_ms"] for r in sel])
    incr = np.array([r["incr_mean_ms"] for r in sel])
    native = sel[0]["native_mean_ms"]
    width, height = sel[0]["width"], sel[0]["height"]

    # Discrete N values, drawn on a CATEGORICAL x-axis (evenly spaced, labelled with
    # the actual N) -- no log scale. amort shape: [tile, N].
    ns = np.array(sorted({int(v) for v in args.ns.split(",") if v.strip()}))
    if ns.size < 2:
        raise SystemExit("Need >=2 distinct N values in --ns.")
    x = np.arange(ns.size)  # categorical positions
    amort = full[:, None] / ns[None, :] + incr[:, None]
    eff = native / amort

    # Upper envelope: best achievable speedup and the tile that achieves it per N.
    best_i = np.argmin(amort, axis=0)
    best_tile = tiles[best_i]
    env = native / amort[best_i, np.arange(ns.size)]

    os.makedirs(args.outdir, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), layout="constrained")

    # Only N is log (it spans decades); speedup stays LINEAR -- a log y-axis made
    # the curves unreadable. Colour each tile consistently across both panels.
    tile_color = {t: plt.cm.viridis(i / max(1, len(tiles) - 1)) for i, t in enumerate(tiles)}

    # (a) effective speedup vs N, one line per *competitive* tile. Tiles whose single
    # incremental update already costs more than a whole native argmax (incr >=
    # native) can never win at any N, so they are dropped -- otherwise their flat
    # near-zero lines just squash the axis.
    ax = axes[0]
    competitive = incr < native
    shown = tiles[competitive] if competitive.any() else tiles
    for t in shown:
        i = int(np.where(tiles == t)[0][0])
        ax.plot(x, eff[i], color=tile_color[t], lw=1.8, marker="o", ms=4, label=f"tile={t}")
    ax.axhline(1.0, color="0.4", ls=":", lw=1.2)
    ax.text(x[-1], 1.0, "break-even ", va="bottom", ha="right", fontsize=8, color="0.4")
    ax.set_xticks(x, [str(n) for n in ns])
    ax.set_ylim(bottom=0)
    ax.set_xlabel("reseed interval N (incremental calls per full recompute)")
    ax.set_ylabel("effective speedup vs native argmax")
    ax.set_title("(a) amortized speedup vs N")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(fontsize=8, loc="upper left", ncol=2)

    # (b) best achievable speedup vs N (the upper envelope over all tiles), with the
    # optimal-tile regions shaded + labelled: shows how much you gain AND which tile
    # delivers it, without a second y-axis.
    ax = axes[1]
    start = 0
    for k in range(1, ns.size + 1):
        if k == ns.size or best_tile[k] != best_tile[start]:
            t = int(best_tile[start])
            # Shade the categorical cells [start, k-1], padded half a cell each side.
            ax.axvspan(x[start] - 0.5, x[k - 1] + 0.5, color=tile_color[t], alpha=0.18)
            ax.text((x[start] + x[k - 1]) / 2, 0.96, f"tile={t}",
                    transform=ax.get_xaxis_transform(), ha="center", va="top",
                    fontsize=8, fontweight="bold")
            start = k
    ax.plot(x, env, color="black", lw=2, marker="o", ms=4, zorder=5)
    ax.axhline(1.0, color="0.4", ls=":", lw=1.2)
    ax.text(x[-1], 1.0, "break-even ", va="bottom", ha="right", fontsize=8, color="0.4")
    ax.set_xticks(x, [str(n) for n in ns])
    ax.set_xlim(x[0] - 0.5, x[-1] + 0.5)
    ax.set_ylim(bottom=0)
    ax.set_xlabel("reseed interval N")
    ax.set_ylabel("best achievable speedup vs native")
    ax.set_title("(b) best speedup + optimal tile vs N")
    ax.grid(True, axis="y", alpha=0.3)

    fig.suptitle(f"Tiled argmax amortization  |  image {width}x{height}, footprint {psf}x{psf}  |  "
                 f"native cub ArgMax = {native:.3f} ms",
                 fontsize=12, fontweight="bold")
    out = os.path.join(args.outdir, f"tiled_argmax_amortized_psf{psf}.png")
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"[ok] {out}  (psf={psf}, {len(tiles)} tiles, N={list(ns)})")


if __name__ == "__main__":
    main()
