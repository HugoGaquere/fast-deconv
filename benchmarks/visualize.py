"""
Visualization script for benchmark results.

Usage:
    python -m benchmarks.visualize results.json
    python -m benchmarks.visualize results.json -o figure.png
    python -m benchmarks.visualize results.json --show
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Style configuration
COLORS = {
    "faster": "#2ecc71",  # Green
    "slower": "#e74c3c",  # Red
    "neutral": "#95a5a6",  # Gray
    "cupy": "#3498db",  # Blue
    "fast_deconv": "#9b59b6",  # Purple
}

CATEGORY_COLORS = {
    "argmax": "#3498db",
    "subtract": "#e67e22",
    "wscms": "#9b59b6",
}


def load_results(path: str | Path) -> list[dict]:
    """Load benchmark results from JSON file."""
    with open(path) as f:
        return json.load(f)


def extract_category(name: str) -> str:
    """Extract category from benchmark name."""
    if "argmax" in name:
        return "argmax"
    elif "subtract_psf" in name or "wscms" in name:
        return "wscms"
    elif "subtract" in name:
        return "subtract"
    return "other"


def plot_speedup_bars(ax: plt.Axes, results: list[dict]) -> None:
    """Plot horizontal bar chart of speedups."""
    # Sort by speedup
    sorted_results = sorted(results, key=lambda x: x["speedup"])

    names = [r["contender"]["name"].replace("fast_deconv:", "") for r in sorted_results]
    speedups = [r["speedup"] for r in sorted_results]
    colors = [COLORS["faster"] if s >= 1 else COLORS["slower"] for s in speedups]

    y_pos = np.arange(len(names))

    bars = ax.barh(y_pos, speedups, color=colors, edgecolor="white", linewidth=0.5)

    # Add speedup labels
    for i, (bar, speedup) in enumerate(zip(bars, speedups)):
        width = bar.get_width()
        label = f"{speedup:.2f}x"
        if speedup >= 1:
            ax.text(width + 0.05, bar.get_y() + bar.get_height() / 2,
                    label, va="center", ha="left", fontsize=8, fontweight="bold")
        else:
            ax.text(width + 0.05, bar.get_y() + bar.get_height() / 2,
                    label, va="center", ha="left", fontsize=8)

    # Reference line at 1.0x
    ax.axvline(x=1.0, color="black", linestyle="--", linewidth=1.5, alpha=0.7)
    ax.text(1.02, len(names) - 0.5, "baseline", fontsize=8, alpha=0.7)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=9)
    ax.set_xlabel("Speedup (higher is better)", fontsize=10)
    ax.set_title("Speedup vs CuPy", fontsize=12, fontweight="bold")
    ax.set_xlim(0, max(speedups) * 1.15)
    ax.grid(axis="x", alpha=0.3)


def plot_category_summary(ax: plt.Axes, results: list[dict]) -> None:
    """Plot category-wise geometric mean speedup."""
    # Group by category
    categories: dict[str, list[float]] = {}
    for r in results:
        name = r["contender"]["name"]
        cat = extract_category(name)
        if cat not in categories:
            categories[cat] = []
        categories[cat].append(r["speedup"])

    # Calculate geometric mean for each category
    cat_names = []
    geo_means = []
    colors = []
    for cat, speedups in sorted(categories.items()):
        cat_names.append(cat)
        geo_mean = np.exp(np.mean(np.log(speedups)))
        geo_means.append(geo_mean)
        colors.append(CATEGORY_COLORS.get(cat, COLORS["neutral"]))

    x_pos = np.arange(len(cat_names))
    bars = ax.bar(x_pos, geo_means, color=colors, edgecolor="white", linewidth=0.5)

    # Add value labels
    for bar, gm in zip(bars, geo_means):
        height = bar.get_height()
        color = COLORS["faster"] if gm >= 1 else COLORS["slower"]
        ax.text(bar.get_x() + bar.get_width() / 2, height + 0.05,
                f"{gm:.2f}x", ha="center", va="bottom", fontsize=10,
                fontweight="bold", color=color)

    ax.axhline(y=1.0, color="black", linestyle="--", linewidth=1.5, alpha=0.7)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(cat_names, fontsize=10)
    ax.set_ylabel("Geometric Mean Speedup", fontsize=10)
    ax.set_title("Speedup by Category", fontsize=12, fontweight="bold")
    ax.set_ylim(0, max(geo_means) * 1.2)
    ax.grid(axis="y", alpha=0.3)


def plot_timing_distribution(ax: plt.Axes, results: list[dict]) -> None:
    """Plot timing distribution with error bars."""
    # Sort by contender median time
    sorted_results = sorted(results, key=lambda x: x["contender"]["median_ms"])

    names = [r["contender"]["name"].replace("fast_deconv:", "") for r in sorted_results]

    # Extract timing data
    cupy_medians = [r["baseline"]["median_ms"] for r in sorted_results]
    cupy_p5 = [r["baseline"]["p5_ms"] for r in sorted_results]
    cupy_p95 = [r["baseline"]["p95_ms"] for r in sorted_results]

    fd_medians = [r["contender"]["median_ms"] for r in sorted_results]
    fd_p5 = [r["contender"]["p5_ms"] for r in sorted_results]
    fd_p95 = [r["contender"]["p95_ms"] for r in sorted_results]

    x_pos = np.arange(len(names))
    width = 0.35

    # CuPy bars with error bars
    cupy_err_low = [m - p5 for m, p5 in zip(cupy_medians, cupy_p5)]
    cupy_err_high = [p95 - m for m, p95 in zip(cupy_medians, cupy_p95)]
    ax.barh(x_pos - width / 2, cupy_medians, width,
            xerr=[cupy_err_low, cupy_err_high],
            color=COLORS["cupy"], label="CuPy", alpha=0.8,
            error_kw={"elinewidth": 1, "capsize": 2})

    # fast_deconv bars with error bars
    fd_err_low = [m - p5 for m, p5 in zip(fd_medians, fd_p5)]
    fd_err_high = [p95 - m for m, p95 in zip(fd_medians, fd_p95)]
    ax.barh(x_pos + width / 2, fd_medians, width,
            xerr=[fd_err_low, fd_err_high],
            color=COLORS["fast_deconv"], label="fast_deconv", alpha=0.8,
            error_kw={"elinewidth": 1, "capsize": 2})

    ax.set_yticks(x_pos)
    ax.set_yticklabels(names, fontsize=8)
    ax.set_xlabel("Median Time (ms) with p5-p95 range", fontsize=10)
    ax.set_title("Execution Time Comparison", fontsize=12, fontweight="bold")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(axis="x", alpha=0.3)


def add_summary_text(fig: plt.Figure, results: list[dict]) -> None:
    """Add summary statistics as text."""
    speedups = [r["speedup"] for r in results]
    geo_mean = np.exp(np.mean(np.log(speedups)))
    faster_count = sum(1 for s in speedups if s > 1)
    total = len(speedups)

    best = max(results, key=lambda x: x["speedup"])
    worst = min(results, key=lambda x: x["speedup"])

    summary = (
        f"Overall: {geo_mean:.2f}x geometric mean speedup\n"
        f"Faster: {faster_count}/{total} benchmarks\n"
        f"Best: {best['contender']['name'].replace('fast_deconv:', '')} ({best['speedup']:.2f}x)\n"
        f"Worst: {worst['contender']['name'].replace('fast_deconv:', '')} ({worst['speedup']:.2f}x)"
    )

    fig.text(0.98, 0.02, summary, fontsize=9, ha="right", va="bottom",
             fontfamily="monospace", bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))


def create_visualization(results: list[dict], output_path: str | None = None, show: bool = False) -> None:
    """Create the complete visualization."""
    # Set up figure
    fig = plt.figure(figsize=(14, 10))

    # Create grid layout
    gs = fig.add_gridspec(2, 2, height_ratios=[2, 1], hspace=0.3, wspace=0.3)

    # Main speedup plot (top, spans both columns)
    ax_speedup = fig.add_subplot(gs[0, :])
    plot_speedup_bars(ax_speedup, results)

    # Category summary (bottom left)
    ax_category = fig.add_subplot(gs[1, 0])
    plot_category_summary(ax_category, results)

    # Timing distribution (bottom right)
    ax_timing = fig.add_subplot(gs[1, 1])
    plot_timing_distribution(ax_timing, results)

    # Add summary text
    add_summary_text(fig, results)

    # Title
    fig.suptitle("fast-deconv vs CuPy Benchmark Results", fontsize=14, fontweight="bold", y=0.98)

    plt.subplots_adjust(top=0.92, bottom=0.08, left=0.15, right=0.95, hspace=0.35, wspace=0.3)

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
        print(f"Figure saved to {output_path}")

    if show:
        plt.show()

    if not output_path and not show:
        # Default: save to same directory as input with .png extension
        plt.savefig("benchmark_results.png", dpi=150, bbox_inches="tight", facecolor="white")
        print("Figure saved to benchmark_results.png")


def main():
    parser = argparse.ArgumentParser(
        description="Visualize benchmark results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "input",
        type=str,
        help="Path to JSON benchmark results file",
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        help="Output path for figure (default: benchmark_results.png)",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display the figure interactively",
    )

    args = parser.parse_args()

    # Load results
    results = load_results(args.input)
    print(f"Loaded {len(results)} benchmark results from {args.input}")

    # Create visualization
    create_visualization(results, args.output, args.show)

    return 0


if __name__ == "__main__":
    sys.exit(main())
