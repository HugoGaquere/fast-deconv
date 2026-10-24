#!/usr/bin/env python3
"""Plot bench timings from the CSVs written by run_sweep_bench.sh (replay_ddmsc --bench --csv).

Renders a horizontal bar chart: one group per image size, one bar per GPU
(the CSV label), each bar the median over the timed runs with a min-max
whisker. Pass every GPU's CSV (or sweep directory) to compare them.

Rows whose run did not do exactly the pinned work (pinned=0) are left out,
with a warning: their timing does not measure the same work.

Examples:
    plot_sweep_bench.py bench_NVIDIA_B200_cycle0_*/ bench_NVIDIA_A100*_cycle0_*/
    plot_sweep_bench.py dalia.csv thetis.csv --metric total_ms --save sweep.png
"""
import argparse
import csv
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt

METRICS = {
    "per_outer_ms": "time per outer iteration",
    "total_ms": "total time (one bench run)",
    "baseline_ms": "baseline time (1 outer iteration, setup included)",
}

# Categorical slots in fixed order, one per GPU; text stays in ink, never the series color.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
INK, INK_2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e2e1dd", "#fcfcfb"


def load(paths: list[Path]) -> list[dict[str, str]]:
    rows = []
    for p in paths:
        path = p / "bench.csv" if p.is_dir() else p
        with path.open(newline="") as f:
            rows += list(csv.DictReader(f))
    return rows


def size_key(r: dict[str, str]) -> tuple[int, int]:
    return int(r["nrow"]), int(r["ncol"])


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("inputs", type=Path, nargs="+", help="bench CSVs, or sweep directories holding bench.csv")
    p.add_argument("--metric", choices=METRICS, default="per_outer_ms", help="column to plot (default: per_outer_ms)")
    p.add_argument("--save", type=Path, help="Write to file instead of showing")
    args = p.parse_args()

    rows = load(args.inputs)
    unpinned = [r for r in rows if r["pinned"] != "1"]
    for r in unpinned:
        print(f"warning: skipping unpinned run {r['run']} of {r['label']} at {r['nrow']}x{r['ncol']}", file=sys.stderr)
    rows = [r for r in rows if r["pinned"] == "1"]
    if not rows:
        print("No pinned rows to plot", file=sys.stderr)
        return 1

    # The bars only compare when every run did the same work per size.
    work = {(r["scale"], r["outer"], r["clean_iter"]) for r in rows}
    if len(work) > 1:
        print(f"warning: runs mix (scale, outer, clean_iter) settings {sorted(work)}; bars are not like for like",
              file=sys.stderr)

    samples: dict[tuple[str, tuple[int, int]], list[float]] = defaultdict(list)
    for r in rows:
        samples[(r["label"], size_key(r))].append(float(r[args.metric]))

    labels = sorted({label for label, _ in samples})
    sizes = sorted({size for _, size in samples}, key=lambda s: s[0] * s[1])
    if len(labels) > len(SERIES):
        print(f"{len(labels)} GPUs but only {len(SERIES)} colors; plot fewer at once", file=sys.stderr)
        return 1

    # Seconds once the values run into the thousands of ms.
    peak = max(max(v) for v in samples.values())
    scale, unit = (1e-3, "s") if peak >= 5000 else (1.0, "ms")

    group_h = 0.8
    bar_h = min(group_h / len(labels), 0.32)
    fig_h = max(3.0, 0.5 + len(sizes) * (len(labels) * bar_h + 0.35))
    fig, ax = plt.subplots(figsize=(9, fig_h), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)

    for i, label in enumerate(labels):
        for j, size in enumerate(sizes):
            v = samples.get((label, size))
            if not v:
                continue
            med, lo, hi = statistics.median(v) * scale, min(v) * scale, max(v) * scale
            # Groups top to bottom by size, GPUs top to bottom inside a group.
            y = j + (i - (len(labels) - 1) / 2) * bar_h
            ax.barh(y, med, height=bar_h * 0.85, color=SERIES[i], label=label if j == 0 else None, zorder=2)
            if len(v) > 1:
                ax.plot([lo, hi], [y, y], color=INK_2, lw=1, zorder=3)
            ax.text(hi, y, f"  {med:.3g} {unit}", va="center", ha="left", color=INK, fontsize=8, zorder=3)

    ax.set_yticks(range(len(sizes)), [f"{r}x{c}" for r, c in sizes])
    ax.invert_yaxis()
    ax.set_xlim(0, ax.get_xlim()[1] * 1.12)  # room for the value labels
    ax.set_xlabel(f"{METRICS[args.metric]} ({unit}) - median, whisker min-max", color=INK_2)
    ax.set_ylabel("image size", color=INK_2)

    n_runs = sorted({len(v) for v in samples.values()})
    runs_txt = f"{n_runs[0]}" if len(n_runs) == 1 else f"{n_runs[0]}-{n_runs[-1]}"
    title = f"replay_ddmsc bench: {METRICS[args.metric]}"
    if len(work) == 1:
        scale_idx, outer, clean_iter = next(iter(work))
        title += f"\nscale {scale_idx}, {outer} outer x {clean_iter} clean iterations, {runs_txt} runs per bar"
    # Legend in its own band above the bars, so it never covers the longest one.
    legend_cols = 2 if len(labels) > 1 else 1
    legend_rows = -(-len(labels) // legend_cols)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.01), ncol=legend_cols, frameon=False, labelcolor=INK,
              fontsize=8, handlelength=1.2, borderaxespad=0)
    ax.set_title(title, color=INK, fontsize=10, loc="left", pad=8 + 15 * legend_rows)

    ax.grid(axis="x", color=GRID, lw=1, zorder=0)
    ax.tick_params(colors=INK_2, length=0)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)

    fig.tight_layout()
    if args.save:
        fig.savefig(args.save, dpi=150, facecolor=SURFACE)
        print(f"wrote {args.save}")
    else:
        plt.show()
    return 0


if __name__ == "__main__":
    sys.exit(main())
