#!/usr/bin/env python3
"""Plot per-cycle WSCMS stats from a CSV produced by example_wscms --csv=PATH.

Renders two figures:
  - timing: bars of wallclock seconds, with minor-iter count on the right axis.
  - peak:   residual peak (final_flux) over cycles, with the stop_flux limit
            drawn for reference.

With --save PATH, both figures are written; the peak plot gets a `_peak`
suffix inserted before the extension (e.g. timings.png -> timings_peak.png).

Examples:
    plot_cycle_timing.py timings.csv
    plot_cycle_timing.py timings.csv --save timings.png
"""
import argparse
import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt


def load(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("csv", type=Path, help="CSV from example_wscms --csv=PATH")
    p.add_argument("--save", type=Path, help="Write to file instead of showing")
    args = p.parse_args()

    rows = load(args.csv)
    if not rows:
        print(f"No rows in {args.csv}", file=sys.stderr)
        return 1

    cycle_ids = [int(r["cycle_id"]) for r in rows]
    elapsed_s = [float(r["elapsed_ms"]) / 1000.0 for r in rows]
    iters = [int(r["total_iterations"]) for r in rows]
    final_flux = [float(r["final_flux"]) for r in rows]
    stop_flux = [float(r["stop_flux"]) for r in rows]
    total_s = sum(elapsed_s)
    total_iters = sum(iters)
    labels = [str(c) for c in cycle_ids]

    width = max(6.0, 0.5 * len(cycle_ids) + 2.0)

    # ---- Figure 1: timing + iterations ----
    fig_t, ax = plt.subplots(figsize=(width, 4.5))
    bars = ax.bar(labels, elapsed_s, color="C0", label="elapsed (s)")
    ax.set_xlabel("cycle id")
    ax.set_ylabel("elapsed (s)", color="C0")
    ax.tick_params(axis="y", labelcolor="C0")
    ax.set_title(
        f"WSCMS per-cycle timing — total {total_s:.2f} s, {total_iters} iters, {len(cycle_ids)} cycle(s)"
    )
    for bar, sec, it in zip(bars, elapsed_s, iters):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{sec:.2f}s\n{it} it",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    ax2 = ax.twinx()
    ax2.plot(labels, iters, color="C3", marker="o", label="iterations")
    ax2.set_ylabel("minor iterations", color="C3")
    ax2.tick_params(axis="y", labelcolor="C3")
    ax2.set_ylim(bottom=0)

    # Headroom so the per-bar text labels don't clip into the title.
    ymax = ax.get_ylim()[1]
    ax.set_ylim(top=ymax * 1.15)

    fig_t.tight_layout()

    # ---- Figure 2: residual peak over cycles ----
    fig_p, axp = plt.subplots(figsize=(width, 4.5))
    axp.plot(labels, final_flux, color="C0", marker="o", label="final residual peak")
    axp.plot(labels, stop_flux, color="C2", linestyle="--", marker="x", label="stop_flux")
    axp.set_xlabel("cycle id")
    axp.set_ylabel("flux")
    axp.set_title(f"WSCMS residual peak over {len(cycle_ids)} cycle(s)")
    if all(f > 0 for f in final_flux + stop_flux):
        axp.set_yscale("log")
    axp.grid(True, which="both", alpha=0.3)
    axp.legend()
    fig_p.tight_layout()

    if args.save:
        save_t = args.save
        save_p = args.save.with_name(args.save.stem + "_peak" + args.save.suffix)
        fig_t.savefig(save_t, dpi=150)
        fig_p.savefig(save_p, dpi=150)
        print(f"Wrote {save_t}")
        print(f"Wrote {save_p}")
    else:
        plt.show()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
