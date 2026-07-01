#!/usr/bin/env python
"""Plot bench_wscms sweep results as image-size scaling curves.

The benchmark is a Cartesian product over many axes. This plots image size on the
x-axis against each metric on the y-axis. The effect of every *other* swept axis
is shown in its own column: within a column we draw one line per value of that
axis while holding all the remaining other axes fixed at their median. So each
cell answers "how does image-size scaling change as this one parameter varies,
with everything else held at a representative baseline?"

The single output image is a grid:
    rows    = metrics (wall time, wall time / iter, compute & data throughput, memory)
    columns = each swept axis other than image size (n_freq, n_scales, ...)
    x       = image size (nrow=ncol), linear by default (--xscale), ticked at sizes
    y       = metric, log by default (--yscale)
    lines   = values of that column's axis

Each axis scale is selectable with --xscale / --yscale {linear,log}. With both on
log the panels become log-log, where a power law N^p is a straight line of slope
p -- handy for reading the wall-time scaling exponent against the N^2 guide.

Usage (always via the project venv):
    .venv/bin/python scripts/plot_bench.py out.csv [--outdir plots] \
        [--xscale linear|log] [--yscale linear|log]
"""
import argparse
import csv
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FuncFormatter, LogLocator, NullLocator

SIZE_AXIS = "nrow"
# Independent config axes other than image size; each swept one becomes a column.
# (psf_nrow is derived from nrow via --psf-frac, so it co-varies and isn't fixed.)
OTHER_AXES = ["n_freq", "n_scales", "n_facet", "n_order", "K", "M"]
AXIS_LABEL = {
    "n_freq": "n_freq", "n_scales": "n_scales", "n_facet": "n_facet",
    "n_order": "n_order", "K": "K (outer)", "M": "M (inner)",
}
# (csv column, axis label, base unit, upgrade, guides). `upgrade` = (coarser_unit,
# scale) lets a row switch to a coarser unit (ms->s, MiB->GiB) once its max reaches
# at least one of it, so the shared per-row log axis reads in a single natural unit
# instead of e.g. "10k" meaning 10k MiB. None = always keep the base unit. `guides`
# overlays an N^2 (linear-in-pixels) scaling reference, anchored at the largest
# size: a metric hugging it scales with pixel count (expected for both wall time
# and memory), data above it at small sizes is fixed overhead, and a steeper-than-
# guide tail flags worse-than-pixel-linear growth (extra compute / over-allocation).
METRICS = [
    ("mean_ms", "wall time", "ms", ("s", 1000.0), True),
    ("ms_per_iter", "wall time / iter", "ms", None, True),
    ("mpix_iter_per_s", "compute throughput", "Mpix.iter/s", None, False),
    ("mvox_iter_per_s", "data throughput", "Mvox.iter/s", None, False),
    ("used_mem_mb", "used memory", "MiB", ("GiB", 1024.0), True),
]
# Hold value for an axis when it is *not* the one being varied (its "median"
# baseline). Auto-computed by mid() unless overridden here. An override must be a
# value actually swept for that axis, and for --mode=ofat must match the baseline
# in src/bench_wscms.cu so the generated cross still lines up.
MEDIAN_OVERRIDE = {"n_freq": 2, "n_scales": 5, "n_facet": 100}
INT_COLS = {
    "nrow", "ncol", "n_freq", "n_scales", "n_facet", "n_order", "psf_nrow",
    "psf_ncol", "K", "M", "total_iters", "n_components", "runs", "warmup",
}


def load(path):
    rows = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            rec = {}
            for k, v in row.items():
                if k == "status":
                    rec[k] = v
                elif k in INT_COLS:
                    rec[k] = int(v)
                else:
                    rec[k] = float(v)
            # Data-volume throughput: voxels = n_freq * pixels, so it is exactly
            # n_freq * the (compute) throughput. Derived here rather than emitted by
            # the benchmark, so it works on existing CSVs.
            rec["mvox_iter_per_s"] = rec["mpix_iter_per_s"] * rec["n_freq"]
            rows.append(rec)
    return rows


def distinct(rows, col):
    return sorted({r[col] for r in rows})


def fmt_size(n):
    """Compact image-size tick label: 2000 -> '2k', 1500 -> '1.5k', 800 -> '800'."""
    if n >= 1000:
        return f"{n / 1000:g}k"
    return str(int(n))


def fmt_val(v):
    """Compact label for a y-value across metrics spanning very different scales
    (sub-1 ms/iter up to tens-of-thousands Mpix.iter/s). Used for both the dense
    log-axis tick labels and the per-panel peak annotation."""
    if v >= 1000:
        return f"{v / 1000:.3g}k"
    if v >= 1:
        return f"{v:.4g}"
    return f"{v:.3g}"


def row_unit(unit, upgrade, vmax):
    """Pick (display unit, divisor) for a metric row from its max value. Upgrades
    to the coarser unit (ms->s, MiB->GiB) once the row reaches >=1 of it, so the
    whole shared axis is labelled in one natural unit. Returns the base unit and a
    divisor of 1.0 when no upgrade applies."""
    if upgrade is not None and vmax >= upgrade[1]:
        return upgrade[0], upgrade[1]
    return unit, 1.0


def scaling_guide(sizes, x0, y0):
    """N^2 (linear-in-pixels) reference curve over side lengths `sizes`, anchored to
    pass through (x0, y0). Anchored at the largest, asymptotic size: wall time
    hugging this curve is pixel-bound scaling, and data sitting above it at small
    sizes is fixed per-run overhead rather than a scaling problem. (An N^2*log N
    guide was dropped -- log N varies too little over a size sweep to separate from
    N^2 on a log axis.)"""
    return [y0 * (x / x0) ** 2 for x in sizes]


def mid(values):
    """Representative element of a sorted distinct list (middle element).

    This must match the OFAT baseline in src/bench_wscms.cu (median_value: sort,
    dedup, take index len//2). If the two diverge, --mode=ofat stops generating
    the rows this plotter selects and every panel silently empties out.
    """
    return values[len(values) // 2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--outdir", default="plots")
    ap.add_argument("--xscale", choices=("linear", "log"), default="linear",
                    help="scale of the image-size x-axis (default linear); log "
                         "makes power-law scaling show as straight lines of "
                         "slope = exponent")
    ap.add_argument("--yscale", choices=("linear", "log"), default="log",
                    help="scale of the metric y-axis (default log, since metrics "
                         "span orders of magnitude)")
    args = ap.parse_args()

    rows = load(args.csv)
    if not rows:
        raise SystemExit("No rows in CSV.")
    ok = [r for r in rows if r["status"] == "ok"]
    if not ok:
        raise SystemExit("No rows with status=ok; nothing to plot.")
    if len(distinct(ok, SIZE_AXIS)) < 2:
        raise SystemExit(f"Image size ({SIZE_AXIS}) was not swept; nothing to plot.")
    os.makedirs(args.outdir, exist_ok=True)

    medians = {a: mid(distinct(ok, a)) for a in OTHER_AXES}
    for a, v in MEDIAN_OVERRIDE.items():
        if a in medians:
            if v not in distinct(ok, a):
                print(f"[warn] MEDIAN_OVERRIDE {a}={v} not present in data; "
                      f"that axis' baseline rows will be empty")
            medians[a] = v
    size_vals = distinct(ok, SIZE_AXIS)  # x-ticks: label exactly the swept sizes
    size_fmt = FuncFormatter(lambda x, _: fmt_size(round(x)))
    logx = args.xscale == "log"
    logy = args.yscale == "log"
    cols = [a for a in OTHER_AXES if len(distinct(ok, a)) > 1]
    if not cols:
        cols = OTHER_AXES[:1]  # nothing else swept: still show the single baseline curve

    nr, nc = len(METRICS), len(cols)
    fig, axes = plt.subplots(nr, nc, figsize=(4.3 * nc + 0.6, 3.5 * nr),
                             sharex=True, sharey="row", squeeze=False,
                             layout="constrained")

    for i, (metric, base_label, base_unit, upgrade, draw_guides) in enumerate(METRICS):
        row_ys = []  # positive y-values across the whole row (shared log y-range)
        panel_maxes = []  # per-column (y, x) peak (or None), for the max annotation
        for j, axis in enumerate(cols):
            ax = axes[i][j]
            vals = distinct(ok, axis)
            drawn = 0
            panel_max = None  # (y, x) of the highest point drawn in this panel
            guide_anchor = None  # baseline line's largest-size point (guide rows)
            guide_fallback = None  # lowest end point, if no baseline line drawable
            for k, v in enumerate(vals):
                # Vary image size; this column's axis = v; all other axes at median.
                sel = [r for r in ok if r[axis] == v
                       and all(r[o] == medians[o] for o in OTHER_AXES if o != axis)]
                sel.sort(key=lambda r: r[SIZE_AXIS])
                if len(sel) < 2:
                    print(f"[warn] {metric}: {axis}={v} (others at median) has "
                          f"{len(sel)} point(s), need >=2 -- line skipped")
                    continue
                xs = [r[SIZE_AXIS] for r in sel]
                ys = [r[metric] for r in sel]
                row_ys += [y for y in ys if y > 0]
                for x, y in zip(xs, ys):
                    if y > 0 and (panel_max is None or y > panel_max[0]):
                        panel_max = (y, x)
                if draw_guides and ys[-1] > 0:
                    # Anchor the N^2 guide to the all-median baseline line at the
                    # largest (asymptotic) size; keep the lowest end point as a
                    # fallback when that baseline line isn't drawable.
                    if v == medians[axis]:
                        guide_anchor = (xs[-1], ys[-1])
                    if guide_fallback is None or ys[-1] < guide_fallback[1]:
                        guide_fallback = (xs[-1], ys[-1])
                # Colour by the value's rank: line axes are ordered numerics, so a
                # sequential map reads naturally, and sampling by index never runs
                # out (a fixed 10-colour cycle silently dropped the 11th+ line).
                frac = k / (len(vals) - 1) if len(vals) > 1 else 0.0
                ax.plot(xs, ys, color=plt.cm.viridis(frac), marker="o", ms=4,
                        label=f"{axis}={v}")
                drawn += 1
            panel_maxes.append(panel_max)
            if drawn == 0:
                print(f"[warn] {metric}: column '{axis}' has no drawable lines "
                      f"(no rows match the median baseline)")
            anchor = guide_anchor if guide_anchor is not None else guide_fallback
            if draw_guides and anchor is not None:
                g_n2 = scaling_guide(size_vals, anchor[0], anchor[1])
                (h_n2,) = ax.plot(size_vals, g_n2, color="0.45", ls="--", lw=1.1,
                                  zorder=1.5, label=r"$\propto N^2$")
                row_ys += [y for y in g_n2 if y > 0]
                ax.legend(handles=[h_n2], fontsize=7, loc="lower right")
            ax.set_yscale("log" if logy else "linear")  # metrics span decades
            if logx:
                ax.set_xscale("log")  # log-log: power laws straighten into lines
            # FixedLocator after set_xscale, which would otherwise reset to the log
            # defaults; label exactly the swept sizes and drop log minor x-ticks.
            ax.xaxis.set_major_locator(FixedLocator(size_vals))
            ax.xaxis.set_major_formatter(size_fmt)
            if logx:
                ax.xaxis.set_minor_locator(NullLocator())
            ax.grid(True, which="both", alpha=0.3)
            if i == 0:
                ax.set_title(AXIS_LABEL.get(axis, axis), fontsize=11)
                if drawn:
                    ax.legend(fontsize=7, loc="best")
            if i == nr - 1:
                ax.set_xlabel("image size (nrow=ncol)" + (", log" if logx else ""))
                ax.tick_params(axis="x", labelrotation=45, labelsize=8)
        # Choose the row's display unit from its max (ms->s, MiB->GiB), then label
        # the y-ticks, peak annotation and y-label in that unit. Deferred to here
        # because the unit depends on the row-wide maximum, known only once every
        # column in the row is plotted.
        unit, scale = row_unit(base_unit, upgrade, max(row_ys) if row_ys else 0.0)
        for j in range(len(cols)):
            ax = axes[i][j]
            # Compact unit-scaled tick labels. Fresh formatter per axis:
            # matplotlib binds each to a single axis.
            fmt = FuncFormatter(lambda y, _, s=scale: fmt_val(y / s))
            ax.yaxis.set_major_formatter(fmt)
            ax.tick_params(axis="y", which="major", labelsize=7)
            if logy:
                # On a log axis, label the decade powers *and* the 2/3/5 minor
                # ticks so values are readable between powers of ten (the default
                # log axis labels only the decades, too coarse to read a peak off).
                ax.yaxis.set_major_locator(LogLocator(base=10))
                ax.yaxis.set_minor_locator(LogLocator(base=10, subs=(2, 3, 5)))
                ax.yaxis.set_minor_formatter(fmt)
                ax.tick_params(axis="y", which="minor", labelsize=6)
            # Call out the panel's peak value, since the shared per-row log axis
            # makes the exact height of the tallest curve hard to read. Right-align
            # when the peak sits at the largest size (the common case) so the label
            # grows leftward into the panel instead of off the right edge.
            panel_max = panel_maxes[j]
            if panel_max is not None:
                ymax, xmax = panel_max
                ha = "right" if xmax == size_vals[-1] else "center"
                ax.annotate(f"max={fmt_val(ymax / scale)} {unit}", xy=(xmax, ymax),
                            xytext=(0, 5), textcoords="offset points",
                            ha=ha, va="bottom", fontsize=7,
                            fontweight="bold", clip_on=False,
                            bbox=dict(boxstyle="round,pad=0.15", fc="white",
                                      ec="0.6", lw=0.5, alpha=0.8))
            if j == 0:
                ax.set_ylabel(f"{base_label} ({unit})", fontsize=9)
        # Pin the shared (sharey="row") y-range from the row's global data
        # min/max. Computing it after every column is plotted avoids the
        # lazy-autoscale / shared-axis clipping where a tall curve in a later
        # column gets cut off by an earlier column's limits. Log can't anchor at
        # 0, so pad multiplicatively; linear reads better anchored at 0.
        if row_ys:
            if logy:
                axes[i][0].set_ylim(min(row_ys) / 1.2, max(row_ys) * 1.2)
            else:
                axes[i][0].set_ylim(0, max(row_ys) * 1.05)

    held = ", ".join(f"{AXIS_LABEL.get(a, a)}={medians[a]}" for a in OTHER_AXES)
    fig.suptitle("Image-size scaling per axis (other axes held at median)\n"
                 f"medians: {held}", fontsize=12, fontweight="bold")
    out = os.path.join(args.outdir, "sweep_image_size.png")
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"[ok] {out}  ({nr}x{nc} panels)")


if __name__ == "__main__":
    main()
