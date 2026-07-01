"""Stage 6 -- numerical fidelity: GPU output vs the DDFacet reference.

Compares the GPU outputs written by the realdata stage (example_wscms
--dump-result) against reference minor-cycle outputs exported from DDFacet,
per cycle: component positions and gains, plus residual-image RMS / max-abs
differences, with a GPU/reference/difference image figure.

Expected reference layout (--ref-dir), matching what --dump-result writes:

    ref_dir/
      cycle_<N>/
        components.csv   # header: row,col,scale,gain[,coeff0..coeffK]
        residual.npy     # float32 (n_freq, nrow, ncol), residual after the
                         # minor cycle

The stage skips cleanly until both sides exist, so the harness can run today
and this comparison activates once the reference exports are available.
"""
from __future__ import annotations

from pathlib import Path

from harness import Ctx, SkipStage, read_csv_rows, write_json


def run(ctx: Ctx) -> dict:
    if not ctx.ref_dir:
        raise SkipStage("no --ref-dir given; reference outputs not available yet "
                        "(see module docstring for the expected layout)")
    gpu_root = ctx.bundle / "realdata" / "gpu_output"
    if not gpu_root.is_dir():
        raise SkipStage("no GPU outputs found -- run the realdata stage with "
                        "--dump-gpu-output (or --ref-dir set) first")
    ref_root = Path(ctx.ref_dir)
    out = ctx.stage_dir("fidelity")

    cycles = sorted(int(d.name.split("_")[1]) for d in gpu_root.glob("cycle_*")
                    if (ref_root / d.name).is_dir())
    if not cycles:
        raise SkipStage(f"no cycle_<N> directories present in both {gpu_root} "
                        f"and {ref_root}")

    per_cycle = {}
    for cid in cycles:
        per_cycle[str(cid)] = compare_cycle(
            gpu_root / f"cycle_{cid}", ref_root / f"cycle_{cid}", out, cid)

    summary = {"cycles_compared": cycles, "per_cycle": per_cycle}
    write_json(out / "metrics.json", summary)
    return summary


def compare_cycle(gpu_dir: Path, ref_dir: Path, out: Path, cid: int) -> dict:
    import numpy as np

    metrics: dict = {}

    # ----- components: aggregate gain per pixel position on both sides -----
    g_comp = read_csv_rows(gpu_dir / "components.csv")
    r_comp = read_csv_rows(ref_dir / "components.csv")

    def gain_map(rows):
        m: dict[tuple[int, int], float] = {}
        for r in rows:
            key = (int(r["row"]), int(r["col"]))
            m[key] = m.get(key, 0.0) + float(r["gain"])
        return m

    gm, rm = gain_map(g_comp), gain_map(r_comp)
    common = set(gm) & set(rm)
    union = set(gm) | set(rm)
    metrics["components"] = {
        "n_gpu": len(g_comp),
        "n_ref": len(r_comp),
        "positions_gpu": len(gm),
        "positions_ref": len(rm),
        "position_jaccard": round(len(common) / len(union), 4) if union else None,
        "total_gain_gpu": round(sum(gm.values()), 6),
        "total_gain_ref": round(sum(rm.values()), 6),
    }
    if common:
        rel = [abs(gm[p] - rm[p]) / max(abs(rm[p]), 1e-12) for p in common]
        metrics["components"]["gain_rel_diff_median"] = round(float(np.median(rel)), 6)
        metrics["components"]["gain_rel_diff_max"] = round(float(np.max(rel)), 6)

    # ----- residual images -----
    g_res = np.load(gpu_dir / "residual.npy")
    r_res = np.load(ref_dir / "residual.npy")
    if g_res.shape != r_res.shape:
        metrics["residual"] = {"error": f"shape mismatch {g_res.shape} vs {r_res.shape}"}
        return metrics
    diff = g_res.astype(np.float64) - r_res.astype(np.float64)
    rms_ref = float(np.sqrt(np.mean(r_res.astype(np.float64) ** 2)))
    rms_diff = float(np.sqrt(np.mean(diff ** 2)))
    metrics["residual"] = {
        "shape": list(g_res.shape),
        "rms_ref": rms_ref,
        "rms_diff": rms_diff,
        "rms_diff_over_rms_ref": rms_diff / rms_ref if rms_ref else None,
        "max_abs_diff": float(np.max(np.abs(diff))),
    }

    _triptych(g_res, r_res, diff, out / f"residual_cycle_{cid}")
    return metrics


def _triptych(g_res, r_res, diff, base) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    from analysis import plots
    plots.setup_style()

    # Mean over frequency for display; shared color scale for GPU/reference.
    g2, r2, d2 = (a.mean(axis=0) if a.ndim == 3 else a for a in (g_res, r_res, diff))
    vmax = float(max(np.abs(g2).max(), np.abs(r2).max()))
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.4))
    for ax, img, title, vm in [(axes[0], g2, "GPU residual", vmax),
                               (axes[1], r2, "reference residual", vmax),
                               (axes[2], d2, "difference", float(np.abs(d2).max()))]:
        im = ax.imshow(img, cmap="RdBu_r", vmin=-vm, vmax=vm, origin="lower")
        ax.set_title(title)
        ax.grid(False)
        fig.colorbar(im, ax=ax, shrink=0.8)
    plots.save_fig(fig, base)
