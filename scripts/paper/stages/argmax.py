"""Stage 2 -- tiled-argmax optimization study via bench_tiled_argmax.

Sweeps tile size x PSF footprint against the native cub::DeviceReduce::ArgMax
baseline (what the clean loop runs today). Produces:
  - per-footprint cost curves vs tile size (the U-shape),
  - a speedup heatmap over (footprint, tile),
  - the amortized effective-speedup curve vs reseed interval N
    (cost model: incr + full/N, matching the bench's breakeven definition).

The image side auto-shrinks when the GPU doesn't comfortably fit the preset
size, so the same preset runs on small cards.
"""
from __future__ import annotations

from harness import Ctx, read_csv_rows, run_cmd, to_num, write_json


def _pick_size(ctx: Ctx, want: int) -> int:
    """Shrink the image side until image + workspace fits in ~60% of VRAM."""
    try:
        total_mib = float(ctx.env["gpu"]["memory.total"])
    except (KeyError, TypeError, ValueError):
        return want
    for size in [want, 16000, 12000, 8000, 4000]:
        need_mib = size * size * 4 * 1.5 / (1024 * 1024)  # image + tile workspace slack
        if need_mib < 0.6 * total_mib:
            return size
    return 2000


def run(ctx: Ctx) -> dict:
    cfg = ctx.preset["argmax"]
    out = ctx.stage_dir("argmax")
    csv_path = out / "tiled_argmax.csv"
    size = _pick_size(ctx, cfg["size"])

    psfs = [p for p in cfg["psfs"] if p <= size]
    run_cmd(ctx, [
        ctx.binary("bench_tiled_argmax"),
        f"--size={size}",
        f"--tiles={','.join(str(t) for t in cfg['tiles'])}",
        f"--psfs={','.join(str(p) for p in psfs)}",
        f"--reps={cfg['reps']}",
        f"--warmup={cfg['warmup']}",
        f"--device={ctx.device}",
        f"--csv={csv_path}",
    ], "argmax_bench")

    rows = [{k: to_num(v) for k, v in r.items()} for r in read_csv_rows(csv_path)]
    summary = analyze(rows, out)
    summary["image_side"] = size
    write_json(out / "summary.json", summary)
    return summary


def analyze(rows: list[dict], out) -> dict:
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.colors import LogNorm
    from matplotlib.ticker import FuncFormatter

    from analysis import plots
    plots.setup_style()

    psfs = sorted({r["psf"] for r in rows})
    tiles = sorted({r["tile"] for r in rows})
    native_ms = rows[0]["native_mean_ms"]

    # --- cost vs tile size, one panel per footprint --------------------------
    fig, axes = plt.subplots(1, len(psfs), figsize=(3.2 * len(psfs), 3.0),
                             squeeze=False, sharey=True)
    for pi, psf in enumerate(psfs):
        ax = axes[0][pi]
        sub = sorted((r for r in rows if r["psf"] == psf), key=lambda r: r["tile"])
        ax.plot([r["tile"] for r in sub], [r["full_mean_ms"] for r in sub],
                marker="s", ms=3, lw=1.2, label="tiled full")
        ax.plot([r["tile"] for r in sub], [r["incr_mean_ms"] for r in sub],
                marker="o", ms=3, lw=1.2, label="tiled incremental")
        ax.axhline(native_ms, ls="--", lw=1, color="gray", label="native (CUB)")
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xticks(tiles)
        ax.set_xticklabels([str(t) for t in tiles])
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
        ax.minorticks_off()
        ax.set_title(f"footprint {psf} px")
        ax.set_xlabel("tile side [px]")
        if pi == 0:
            ax.set_ylabel("time per call [ms]")
            ax.legend()
    plots.save_fig(fig, out / "argmax_tile_sweep")

    # --- speedup heatmap (footprint x tile) ----------------------------------
    grid = np.full((len(psfs), len(tiles)), np.nan)
    for r in rows:
        grid[psfs.index(r["psf"]), tiles.index(r["tile"])] = r["speedup_incr"]
    fig, ax = plt.subplots(figsize=(1.0 + 0.8 * len(tiles), 1.0 + 0.6 * len(psfs)))
    im = ax.imshow(grid, aspect="auto", cmap="viridis",
                   norm=LogNorm(vmin=max(np.nanmin(grid), 0.1), vmax=np.nanmax(grid)))
    ax.set_xticks(range(len(tiles)), [str(t) for t in tiles])
    ax.set_yticks(range(len(psfs)), [str(p) for p in psfs])
    ax.set_xlabel("tile side [px]")
    ax.set_ylabel("PSF footprint [px]")
    ax.set_title("incremental argmax speedup vs native")
    ax.grid(False)
    for i in range(len(psfs)):
        for j in range(len(tiles)):
            if not np.isnan(grid[i, j]):
                ax.text(j, i, f"{grid[i, j]:.0f}", ha="center", va="center",
                        fontsize=7, color="white")
    fig.colorbar(im, ax=ax, label="speedup (x)")
    plots.save_fig(fig, out / "argmax_speedup_heatmap")

    # --- amortized effective speedup vs reseed interval ----------------------
    # Per minor iteration the tiled scheme pays incr plus a full reseed every N
    # iterations: eff(N) = native / (incr + full/N).
    psf_big = psfs[-1]
    sub = sorted((r for r in rows if r["psf"] == psf_big),
                 key=lambda r: r["speedup_incr"], reverse=True)[:4]
    N = np.logspace(0, 5, 200)
    fig, ax = plt.subplots(figsize=(4.2, 3.2))
    for r in sorted(sub, key=lambda r: r["tile"]):
        eff = native_ms / (r["incr_mean_ms"] + r["full_mean_ms"] / N)
        ax.plot(N, eff, lw=1.2, label=f"tile {r['tile']}")
    ax.axhline(1.0, ls="--", lw=1, color="gray")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("reseed interval N [iterations]")
    ax.set_ylabel("effective speedup vs native")
    ax.set_title(f"amortized speedup (footprint {psf_big} px)")
    ax.legend()
    plots.save_fig(fig, out / "argmax_amortized")

    per_psf = {}
    for psf in psfs:
        best = max((r for r in rows if r["psf"] == psf),
                   key=lambda r: r["speedup_incr"])
        per_psf[str(psf)] = {
            "best_tile": best["tile"],
            "incr_ms": round(best["incr_mean_ms"], 4),
            "full_ms": round(best["full_mean_ms"], 4),
            "speedup_incr": round(best["speedup_incr"], 1),
            "breakeven_n": round(best["breakeven_n"], 1),
        }
    return {"native_ms": round(native_ms, 4), "per_footprint": per_psf}
