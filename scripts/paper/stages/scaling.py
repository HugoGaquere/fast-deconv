"""Stage 1 -- scaling & throughput sweep via bench_ddmsc (synthetic, fixed K*M work).

Runs the OFAT parameter sweep, then renders image-size scaling panels (one
column per swept axis, one row per metric, log-log with an N^2 guide), fits the
wall-time and memory power-law exponents on the baseline curve, and records the
headline throughput numbers at the largest completed size.

Configs that exceed device memory are skipped by the bench itself
(status=skipped_oom), so the same preset runs unchanged on smaller GPUs.
"""
from __future__ import annotations

from harness import Ctx, read_csv_rows, run_cmd, to_num, write_json

AXES = ["n_freq", "n_scales", "n_facet", "n_order", "K", "M"]

# (csv column, row label, scale factor, draw N^2 guide)
METRIC_ROWS = [
    ("mean_ms", "wall time [s]", 1e-3, True),
    ("ms_per_iter", "time per minor iter [ms]", 1.0, True),
    ("mpix_iter_per_s", "throughput [Mpix·iter/s]", 1.0, False),
    ("used_mem_mb", "device memory [GiB]", 1.0 / 1024.0, True),
]


def _lst(v) -> str:
    return ",".join(str(x) for x in v)


def run(ctx: Ctx) -> dict:
    cfg = ctx.preset["scaling"]
    out = ctx.stage_dir("scaling")
    csv_path = out / "bench.csv"

    run_cmd(ctx, [
        ctx.binary("bench_ddmsc"),
        f"--mode={cfg['mode']}",
        f"--sizes={_lst(cfg['sizes'])}",
        f"--nfreq={_lst(cfg['nfreq'])}",
        f"--nscales={_lst(cfg['nscales'])}",
        f"--nfacet={_lst(cfg['nfacet'])}",
        f"--norder={_lst(cfg['norder'])}",
        f"--psf-frac={_lst(cfg['psf_frac'])}",
        f"--K={_lst(cfg['K'])}",
        f"--M={_lst(cfg['M'])}",
        f"--runs={cfg['runs']}",
        f"--warmup={cfg['warmup']}",
        f"--device={ctx.device}",
        f"--csv={csv_path}",
        "--force",
    ], "scaling_bench")

    rows = [{k: to_num(v) for k, v in r.items()} for r in read_csv_rows(csv_path)]
    summary = analyze(rows, out)
    write_json(out / "summary.json", summary)
    return summary


# --------------------------------------------------------------------------- #
#  Analysis (also reused by aggregate for the cross-GPU overlay)
# --------------------------------------------------------------------------- #

# Hold-at overrides matching the OFAT baseline pinned in src/bench_ddmsc.cu
# (n_freq=2, n_scales=5, n_facet=100). Any axis not listed holds at the median of
# its distinct values.
PINNED_BASELINES = {"n_freq": 2, "n_scales": 5, "n_facet": 100}


def axis_baselines(rows: list[dict]) -> dict:
    """Hold-at value per axis: the pinned baseline where one is set (and present
    in the data), else the median of distinct values, mirroring the bench's OFAT
    mode so the selected cross lines up with the generated rows."""
    base = {}
    for ax in AXES:
        vals = sorted({r[ax] for r in rows})
        pin = PINNED_BASELINES.get(ax)
        base[ax] = pin if (pin is not None and pin in vals) else vals[len(vals) // 2]
    return base


def select(rows: list[dict], fixed: dict) -> list[dict]:
    out = [r for r in rows if all(r[k] == v for k, v in fixed.items())]
    return sorted(out, key=lambda r: r["nrow"])


def baseline_curve(rows: list[dict]) -> list[dict]:
    ok = [r for r in rows if r["status"] == "ok"]
    return select(ok, axis_baselines(ok)) if ok else []


def fit_exponent(xs, ys):
    """Power-law exponent of y ~ x^p fitted over the upper half of the sizes
    (small sizes carry fixed overhead that flattens the slope)."""
    import numpy as np
    pts = [(x, y) for x, y in zip(xs, ys) if x > 0 and y > 0]
    if len(pts) < 3:
        return None
    xs = np.array([p[0] for p in pts], float)
    ys = np.array([p[1] for p in pts], float)
    keep = xs >= np.median(xs)
    if keep.sum() < 2:
        return None
    return float(np.polyfit(np.log(xs[keep]), np.log(ys[keep]), 1)[0])


def analyze(all_rows: list[dict], out) -> dict:
    import matplotlib.pyplot as plt

    from analysis import plots
    plots.setup_style()

    ok = [r for r in all_rows if r["status"] == "ok"]
    if not ok:
        return {"note": "no successful configs", "n_configs": len(all_rows)}
    base = axis_baselines(ok)
    baseline = select(ok, base)

    # One column per axis that was actually swept (K is held at baseline, not
    # shown: its sweep mirrors M under the fixed K*M synthetic work).
    cols = [ax for ax in AXES
            if ax != "K" and len({r[ax] for r in ok}) > 1] or [AXES[0]]
    sizes = sorted({r["nrow"] for r in ok})  # x-ticks: exactly the swept sizes

    fig, axes2d = plt.subplots(len(METRIC_ROWS), len(cols),
                               figsize=(3.4 * len(cols), 2.7 * len(METRIC_ROWS)),
                               squeeze=False, sharex=True)
    for ci, ax_name in enumerate(cols):
        values = sorted({r[ax_name] for r in ok})
        for ri, (metric, label, scale, guide) in enumerate(METRIC_ROWS):
            cell = axes2d[ri][ci]
            panel_max = None  # (y, x) of the highest plotted point
            for v in values:
                fixed = {a: base[a] for a in AXES if a != ax_name}
                fixed[ax_name] = v
                rows_v = select(ok, fixed)
                if not rows_v:
                    continue
                xs = [r["nrow"] for r in rows_v]
                ys = [r[metric] * scale for r in rows_v]
                cell.plot(xs, ys, marker="o", ms=3, lw=1.2, label=f"{ax_name}={v}")
                for x, y in zip(xs, ys):
                    if y > 0 and (panel_max is None or y > panel_max[0]):
                        panel_max = (y, x)
            if panel_max is not None:
                plots.annotate_max(cell, panel_max[1], panel_max[0], sizes,
                                   plots.unit_of(label))
            if guide and len(baseline) >= 2:
                x0, y0 = baseline[-1]["nrow"], baseline[-1][metric] * scale
                xg = sorted({r["nrow"] for r in ok})
                cell.plot(xg, [y0 * (x / x0) ** 2 for x in xg],
                          ls="--", lw=1, color="gray", label="$N^2$")
            cell.set_xscale("log")
            cell.set_yscale("log")
            plots.scaling_axes(cell, sizes)
            if ri == 0:
                cell.set_title(ax_name)
                cell.legend()
            if ri == len(METRIC_ROWS) - 1:
                cell.set_xlabel("image side N [px]")
            if ci == 0:
                cell.set_ylabel(label)
    plots.save_fig(fig, out / "scaling_panels")

    summary = {
        "n_configs": len(all_rows),
        "n_ok": len(ok),
        "n_skipped_oom": sum(1 for r in all_rows if r["status"] == "skipped_oom"),
        "n_other_status": sum(
            1 for r in all_rows if r["status"] not in ("ok", "skipped_oom")),
        "baseline": base,
    }
    if baseline:
        xs = [r["nrow"] for r in baseline]
        summary["exponent_time_vs_side"] = fit_exponent(
            xs, [r["mean_ms"] for r in baseline])
        summary["exponent_mem_vs_side"] = fit_exponent(
            xs, [r["used_mem_mb"] for r in baseline])
        largest = baseline[-1]
        summary["at_largest_baseline_size"] = {
            "nrow": largest["nrow"],
            "wall_s": round(largest["mean_ms"] / 1e3, 3),
            "ms_per_iter": round(largest["ms_per_iter"], 4),
            "mpix_iter_per_s": round(largest["mpix_iter_per_s"], 1),
            "used_mem_mb": round(largest["used_mem_mb"], 1),
        }
    return summary
