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


def simplify_name(name: str) -> str:
    """Simplify benchmark name for display."""
    name = name.replace("fast_deconv:", "")
    name = name.replace("subtract_psf_from_dirty", "psf_subtract")
    name = name.replace("_strided", " (strided)")
    name = name.replace("_inplace", " (inplace)")
    name = name.replace("_masked", " (masked)")
    name = name.replace("_abs", " (abs)")
    return name


def plot_speedup_bars(ax: plt.Axes, results: list[dict]) -> None:
    """Plot horizontal bar chart of speedups, grouped by category."""
    # Group by category (order: best performing first)
    categories_order = ["wscms", "subtract", "argmax"]
    grouped: dict[str, list[dict]] = {cat: [] for cat in categories_order}

    for r in results:
        cat = extract_category(r["contender"]["name"])
        if cat in grouped:
            grouped[cat].append(r)

    # Sort within each group by speedup (descending)
    for cat in grouped:
        grouped[cat].sort(key=lambda x: x["speedup"], reverse=True)

    # Build flat list with group separators
    names = []
    speedups = []
    colors = []
    group_positions = []  # Track where each group starts
    y_pos_list = []

    current_y = 0
    max_speedup = 0

    for cat in categories_order:
        if not grouped[cat]:
            continue

        group_positions.append((current_y, cat))

        for r in grouped[cat]:
            name = simplify_name(r["contender"]["name"])
            speedup = r["speedup"]

            names.append(name)
            speedups.append(speedup)
            colors.append(CATEGORY_COLORS.get(cat, COLORS["neutral"]))
            y_pos_list.append(current_y)
            max_speedup = max(max_speedup, speedup)
            current_y += 1

        # Add spacing between groups
        current_y += 0.5

    y_pos = np.array(y_pos_list)

    # Draw bars
    bars = ax.barh(y_pos, speedups, color=colors, edgecolor="white", linewidth=0.5, alpha=0.85)

    # Add speedup labels
    for bar, speedup in zip(bars, speedups):
        width = bar.get_width()
        label = f"{speedup:.2f}x"
        fontweight = "bold" if speedup >= 1 else "normal"
        ax.text(width + 0.05, bar.get_y() + bar.get_height() / 2,
                label, va="center", ha="left", fontsize=8, fontweight=fontweight)

    # Add category labels and separators
    for start_y, cat in group_positions:
        # Category label on left
        ax.text(-0.15, start_y + len(grouped[cat]) / 2 - 0.5, cat.upper(),
                va="center", ha="right", fontsize=10, fontweight="bold",
                color=CATEGORY_COLORS.get(cat, "black"), transform=ax.get_yaxis_transform())

    # Reference line at 1.0x
    ax.axvline(x=1.0, color="black", linestyle="--", linewidth=1.5, alpha=0.7)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=9)
    ax.set_xlabel("Speedup (higher is better)", fontsize=10)
    ax.set_title("Speedup vs CuPy (grouped by category)", fontsize=12, fontweight="bold")
    ax.set_xlim(0, max_speedup * 1.2)
    ax.grid(axis="x", alpha=0.3)

    # Adjust y-axis limits to show all bars
    ax.set_ylim(-0.5, current_y - 0.5)


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
    """Plot timing distribution with error bars, grouped by category."""
    # Group by category (same order as speedup chart)
    categories_order = ["wscms", "subtract", "argmax"]
    grouped: dict[str, list[dict]] = {cat: [] for cat in categories_order}

    for r in results:
        cat = extract_category(r["contender"]["name"])
        if cat in grouped:
            grouped[cat].append(r)

    # Sort within each group by contender median time
    for cat in grouped:
        grouped[cat].sort(key=lambda x: x["contender"]["median_ms"], reverse=True)

    # Build flat list with group separators
    names = []
    cupy_medians = []
    cupy_p5 = []
    cupy_p95 = []
    fd_medians = []
    fd_p5 = []
    fd_p95 = []
    y_pos_list = []

    current_y = 0

    for cat in categories_order:
        if not grouped[cat]:
            continue

        for r in grouped[cat]:
            name = simplify_name(r["contender"]["name"])
            names.append(name)
            cupy_medians.append(r["baseline"]["median_ms"])
            cupy_p5.append(r["baseline"]["p5_ms"])
            cupy_p95.append(r["baseline"]["p95_ms"])
            fd_medians.append(r["contender"]["median_ms"])
            fd_p5.append(r["contender"]["p5_ms"])
            fd_p95.append(r["contender"]["p95_ms"])
            y_pos_list.append(current_y)
            current_y += 1

        # Add spacing between groups
        current_y += 0.5

    y_pos = np.array(y_pos_list)
    width = 0.35

    # CuPy bars with error bars
    cupy_err_low = [m - p5 for m, p5 in zip(cupy_medians, cupy_p5)]
    cupy_err_high = [p95 - m for m, p95 in zip(cupy_medians, cupy_p95)]
    ax.barh(y_pos - width / 2, cupy_medians, width,
            xerr=[cupy_err_low, cupy_err_high],
            color=COLORS["cupy"], label="CuPy", alpha=0.8,
            error_kw={"elinewidth": 1, "capsize": 2})

    # fast_deconv bars with error bars
    fd_err_low = [m - p5 for m, p5 in zip(fd_medians, fd_p5)]
    fd_err_high = [p95 - m for m, p95 in zip(fd_medians, fd_p95)]
    ax.barh(y_pos + width / 2, fd_medians, width,
            xerr=[fd_err_low, fd_err_high],
            color=COLORS["fast_deconv"], label="fast_deconv", alpha=0.8,
            error_kw={"elinewidth": 1, "capsize": 2})

    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=8)
    ax.set_xlabel("Median Time (ms)", fontsize=10)
    ax.set_title("Execution Time Comparison", fontsize=12, fontweight="bold")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(axis="x", alpha=0.3)
    ax.set_ylim(-0.5, current_y - 0.5)


def format_shape(metadata: dict) -> str:
    """Format shape information from metadata."""
    if "shape" in metadata:
        shape = metadata["shape"]
        if isinstance(shape, list):
            return "×".join(str(s) for s in shape)
        return str(shape)
    elif "n_channels" in metadata:
        return f"{metadata['n_channels']}×{metadata.get('n_pol', 1)}×{metadata['height']}×{metadata['width']}"
    elif "total_elements" in metadata:
        return f"{metadata['total_elements']:,} elements"
    return "N/A"


def print_details_table(results: list[dict]) -> None:
    """Print a detailed table of benchmark results to console."""
    # Group by category
    categories_order = ["argmax", "subtract", "wscms"]
    grouped: dict[str, list[dict]] = {cat: [] for cat in categories_order}

    for r in results:
        cat = extract_category(r["contender"]["name"])
        if cat in grouped:
            grouped[cat].append(r)

    # Sort within each group by speedup
    for cat in grouped:
        grouped[cat].sort(key=lambda x: x["speedup"], reverse=True)

    # Print header
    print("\n" + "=" * 100)
    print(f"{'Benchmark':<30} {'Shape':<20} {'CuPy (ms)':<12} {'fast_deconv (ms)':<16} {'Speedup':<10}")
    print("=" * 100)

    for cat in categories_order:
        if not grouped[cat]:
            continue

        print(f"\n{cat.upper()}")
        print("-" * 100)

        for r in grouped[cat]:
            name = simplify_name(r["contender"]["name"])
            metadata = r["contender"].get("metadata", {})
            shape = format_shape(metadata)
            cupy_time = r["baseline"]["median_ms"]
            fd_time = r["contender"]["median_ms"]
            speedup = r["speedup"]

            speedup_str = f"{speedup:.2f}x"
            if speedup >= 1:
                speedup_str = f"\033[92m{speedup_str}\033[0m"  # Green
            else:
                speedup_str = f"\033[91m{speedup_str}\033[0m"  # Red

            print(f"{name:<30} {shape:<20} {cupy_time:<12.4f} {fd_time:<16.4f} {speedup_str:<10}")

    print("\n" + "=" * 100)


def create_table_figure(results: list[dict]) -> plt.Figure:
    """Create a separate figure with a detailed results table."""
    # Group by category
    categories_order = ["argmax", "subtract", "wscms"]
    grouped: dict[str, list[dict]] = {cat: [] for cat in categories_order}

    for r in results:
        cat = extract_category(r["contender"]["name"])
        if cat in grouped:
            grouped[cat].append(r)

    # Sort within each group by speedup
    for cat in grouped:
        grouped[cat].sort(key=lambda x: x["speedup"], reverse=True)

    # Build table data
    table_data = []
    row_colors = []

    for cat in categories_order:
        if not grouped[cat]:
            continue

        # Category header row
        table_data.append([cat.upper(), "", "", "", "", ""])
        row_colors.append(CATEGORY_COLORS.get(cat, "#cccccc"))

        for r in grouped[cat]:
            name = simplify_name(r["contender"]["name"])
            metadata = r["contender"].get("metadata", {})
            shape = format_shape(metadata)
            cupy_time = f"{r['baseline']['median_ms']:.4f}"
            fd_time = f"{r['contender']['median_ms']:.4f}"
            speedup = r["speedup"]
            speedup_str = f"{speedup:.2f}x"

            table_data.append([f"  {name}", shape, cupy_time, fd_time, speedup_str,
                              "✓" if speedup >= 1 else "✗"])
            row_colors.append("#e8f5e9" if speedup >= 1 else "#ffebee")

    # Create figure
    fig, ax = plt.subplots(figsize=(14, len(table_data) * 0.4 + 1))
    ax.axis("off")

    # Create table
    col_labels = ["Benchmark", "Shape", "CuPy (ms)", "fast_deconv (ms)", "Speedup", ""]
    table = ax.table(
        cellText=table_data,
        colLabels=col_labels,
        loc="center",
        cellLoc="left",
        colWidths=[0.25, 0.18, 0.14, 0.16, 0.12, 0.05],
    )

    # Style table
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.5)

    # Style header
    for j in range(len(col_labels)):
        table[(0, j)].set_facecolor("#2c3e50")
        table[(0, j)].set_text_props(color="white", fontweight="bold")

    # Style rows
    for i, color in enumerate(row_colors):
        for j in range(len(col_labels)):
            cell = table[(i + 1, j)]
            if table_data[i][1] == "":  # Category header
                cell.set_facecolor(color)
                cell.set_text_props(color="white", fontweight="bold")
            else:
                cell.set_facecolor(color)

    fig.suptitle("Benchmark Details", fontsize=14, fontweight="bold")
    plt.tight_layout()

    return fig


def add_summary_text(fig: plt.Figure, results: list[dict]) -> None:
    """Add summary statistics as text."""
    speedups = [r["speedup"] for r in results]
    geo_mean = np.exp(np.mean(np.log(speedups)))
    faster_count = sum(1 for s in speedups if s > 1)
    total = len(speedups)

    best = max(results, key=lambda x: x["speedup"])
    worst = min(results, key=lambda x: x["speedup"])

    summary = (
        f"Overall: {geo_mean:.2f}x geo mean | "
        f"Faster: {faster_count}/{total} | "
        f"Best: {simplify_name(best['contender']['name'])} ({best['speedup']:.2f}x) | "
        f"Worst: {simplify_name(worst['contender']['name'])} ({worst['speedup']:.2f}x)"
    )

    # Place at the bottom center, below all plots
    fig.text(0.5, 0.01, summary, fontsize=9, ha="center", va="bottom",
             fontfamily="monospace", bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.7))


def create_visualization(results: list[dict], output_path: str | None = None, show: bool = False) -> None:
    """Create the complete visualization."""
    # Set up figure
    fig = plt.figure(figsize=(14, 12))

    # Create grid layout: speedup + category on top row, timing at bottom
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1], width_ratios=[3, 1], hspace=0.25, wspace=0.25)

    # Main speedup plot (top left, larger)
    ax_speedup = fig.add_subplot(gs[0, 0])
    plot_speedup_bars(ax_speedup, results)

    # Category summary (top right, smaller)
    ax_category = fig.add_subplot(gs[0, 1])
    plot_category_summary(ax_category, results)

    # Timing distribution (bottom, spans both columns)
    ax_timing = fig.add_subplot(gs[1, :])
    plot_timing_distribution(ax_timing, results)

    # Add summary text
    add_summary_text(fig, results)

    # Title
    fig.suptitle("fast-deconv vs CuPy Benchmark Results", fontsize=14, fontweight="bold", y=0.98)

    plt.subplots_adjust(top=0.94, bottom=0.06, left=0.12, right=0.95, hspace=0.3, wspace=0.2)

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
    parser.add_argument(
        "--table",
        action="store_true",
        help="Print detailed table to console",
    )
    parser.add_argument(
        "--table-image",
        type=str,
        metavar="PATH",
        help="Save detailed table as image",
    )

    args = parser.parse_args()

    # Load results
    results = load_results(args.input)
    print(f"Loaded {len(results)} benchmark results from {args.input}")

    # Print table if requested
    if args.table:
        print_details_table(results)

    # Save table image if requested
    if args.table_image:
        fig = create_table_figure(results)
        fig.savefig(args.table_image, dpi=150, bbox_inches="tight", facecolor="white")
        print(f"Table saved to {args.table_image}")
        plt.close(fig)

    # Create main visualization (unless only table was requested)
    if not (args.table and not args.output and not args.show and not args.table_image):
        create_visualization(results, args.output, args.show)

    return 0


if __name__ == "__main__":
    sys.exit(main())
