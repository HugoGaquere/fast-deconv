"""Stage 5 -- real-data cycle timings + convergence via example_ddmsc.

Requires --dump-dir pointing at a FastDDFacet dump_ref export (init/ +
cycle_<N>/ subdirectories). Produces per-cycle wall time, iteration counts,
ms per minor iteration, and the convergence plot (residual peak vs the stop
threshold).

With --dump-gpu-output (or when --ref-dir is set), example_ddmsc additionally
writes each cycle's component list and final residual under
realdata/gpu_output/ for the fidelity stage. The residual is
n_freq * nrow * ncol float32 -- several GiB for survey-size images -- which is
why this is opt-in.
"""
from __future__ import annotations

import re
from pathlib import Path

from harness import Ctx, SkipStage, read_csv_rows, run_cmd, to_num, write_json

_INITIAL_FLUX_RE = re.compile(r"initial pak_flux=([-\d.eE+]+)")


def _initial_fluxes(out) -> list[float]:
    """Per-cycle initial peak flux parsed, in cycle order, from the realdata run
    log -- example_ddmsc logs one 'initial pak_flux=' line per cycle but does not
    write it to the CSV. Empty when the log is absent."""
    log = Path(out).parent / "logs" / "realdata_run.log"
    if not log.exists():
        return []
    return [float(m) for m in _INITIAL_FLUX_RE.findall(log.read_text())]


def run(ctx: Ctx) -> dict:
    if not ctx.dump_dir:
        raise SkipStage("no --dump-dir given (needs a FastDDFacet dump_ref export)")
    out = ctx.stage_dir("realdata")
    csv_path = out / "cycles.csv"

    cmd = [
        ctx.binary("example_ddmsc"),
        ctx.dump_dir,
        f"--cycles={ctx.cycles}",
        f"--device={ctx.device}",
        f"--csv={csv_path}",
    ]
    if ctx.dump_gpu_output or ctx.ref_dir:
        cmd.append(f"--dump-result={out / 'gpu_output'}")
    if ctx.force_auto_mask_last:
        cmd.append("--force-auto-mask-last")
    run_cmd(ctx, cmd, "realdata_run")

    rows = [{k: to_num(v) for k, v in r.items()} for r in read_csv_rows(csv_path)]
    summary = analyze(rows, out)
    write_json(out / "summary.json", summary)
    return summary


def analyze(rows: list[dict], out) -> dict:
    import matplotlib.pyplot as plt

    from analysis import plots
    plots.setup_style()

    cycles = [r["cycle_id"] for r in rows]
    elapsed_s = [r["elapsed_ms"] / 1e3 for r in rows]
    iters = [r["total_iterations"] for r in rows]
    xpos = list(range(len(cycles)))

    # Wall time per cycle (bars) with the minor-iteration count overlaid.
    fig, ax = plt.subplots(figsize=(4.6, 3.2))
    ax.set_axisbelow(True)  # grid behind the bars
    bars = ax.bar(xpos, elapsed_s, color="#3b738f", label="wall time")
    # label each bar with its wall time and minor-iteration count
    headroom = max(elapsed_s) * 0.02 if elapsed_s else 0
    for bar, s, it in zip(bars, elapsed_s, iters):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + headroom,
                f"{s:.1f} s\n{int(it)} it", ha="center", va="bottom", fontsize=7)
    ax.set_ylim(top=max(elapsed_s) * 1.18 if elapsed_s else 1)
    ax.set_xlabel("major cycle")
    ax.set_ylabel("minor-cycle wall time [s]")
    # iteration counts are already on the bar labels, so use the second axis for
    # the residual peak after each cycle. Anchor it at a leading "start" point --
    # the dirty-image peak before any cleaning (first cycle's initial pak_flux from
    # the run log) -- to show where convergence came down from. Log-scaled for the
    # wide range between the start peak and the residuals.
    initial = _initial_fluxes(out)
    ax2 = ax.twinx()
    if initial:
        ax2.plot([-1] + xpos, [initial[0]] + [r["final_flux"] for r in rows],
                 color="#d1605e", marker="o", ms=4, alpha=0.8)
        ax.set_xticks([-1] + xpos, ["start"] + [str(c) for c in cycles])
        ax.set_xlim(-1.7, len(cycles) - 0.3)
    else:
        ax2.plot(xpos, [r["final_flux"] for r in rows],
                 color="#d1605e", marker="o", ms=4, alpha=0.8)
        ax.set_xticks(xpos, [str(c) for c in cycles])
    ax2.set_yscale("log")
    ax2.set_ylabel("peak flux [Jy]", color="#d1605e")
    ax2.tick_params(axis="y", colors="#d1605e")
    ax2.spines["right"].set_color("#d1605e")
    ax2.grid(False)
    ax.set_title("Per-major-cycle runtime and peak-flux convergence")
    plots.save_fig(fig, out / "cycle_timing")

    # Convergence: residual peak after each cycle vs the stop threshold.
    fig, ax = plt.subplots(figsize=(4.6, 3.2))
    ax.set_axisbelow(True)  # grid behind the curves
    ax.plot([str(c) for c in cycles], [r["final_flux"] for r in rows],
            marker="o", ms=4, color="#3b738f", label="residual peak")
    ax.plot([str(c) for c in cycles], [r["stop_flux"] for r in rows],
            marker="s", ms=4, ls="--", color="#d1605e", label="stop threshold")
    ax.set_yscale("log")
    ax.set_xlabel("major cycle")
    ax.set_ylabel("flux [Jy]")
    ax.set_title("convergence on real data")
    ax.legend()
    plots.save_fig(fig, out / "convergence")

    total_s = sum(elapsed_s)
    total_iters = sum(iters)
    return {
        "n_cycles": len(rows),
        "total_wall_s": round(total_s, 2),
        "total_minor_iterations": total_iters,
        "mean_ms_per_iter": round(1e3 * total_s / total_iters, 4) if total_iters else None,
        "total_components": sum(r["n_components"] for r in rows),
    }
