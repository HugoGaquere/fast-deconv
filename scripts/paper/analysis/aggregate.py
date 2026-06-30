"""Cross-bundle aggregation: overlay scaling curves and merge headline tables.

Takes N bundle directories (each produced by `paper.py run`, typically one per
GPU -- or two per GPU with different --label for an A/B such as the
tiled-argmax integration) and emits paper-ready comparison figures plus CSV /
Markdown / LaTeX tables. Runs anywhere; no GPU needed.
"""
from __future__ import annotations

from pathlib import Path

from harness import read_csv_rows, read_json, to_num
from stages.scaling import (AXES, METRIC_ROWS, axis_baselines, baseline_curve,
                            select)


def _load_bundle(path: Path) -> dict:
    b = {"path": path, "env": {}, "summary": {}}
    if (path / "env.json").exists():
        b["env"] = read_json(path / "env.json")
    if (path / "summary.json").exists():
        b["summary"] = read_json(path / "summary.json")
    gpu = b["env"].get("gpu", {}).get("name", path.name)
    label = b["env"].get("label", "")
    b["label"] = f"{gpu} ({label})" if label else gpu
    scaling_csv = path / "scaling" / "bench.csv"
    if scaling_csv.exists():
        rows = [{k: to_num(v) for k, v in r.items()}
                for r in read_csv_rows(scaling_csv)]
        b["baseline"] = baseline_curve(rows)
        b["rows"] = [r for r in rows if r["status"] == "ok"]
    return b


def run(bundle_paths, outdir) -> dict:
    import matplotlib.pyplot as plt
    import numpy as np

    from analysis import plots, tables
    plots.setup_style()

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    bundles = [_load_bundle(Path(p)) for p in bundle_paths]

    # ----- cross-GPU scaling: two standalone figures -----
    # The portability headline (baseline wall-time and throughput overlaid across
    # GPUs) and the per-GPU OFAT parameter sweep used to read different things, so
    # they are now separate figures rather than stacked rows of one grid. K is held
    # at baseline, not shown (its sweep mirrors M under the fixed K*M synthetic
    # work), matching the per-bundle panels.
    with_scaling = [b for b in bundles if b.get("baseline")]
    sweep = [b for b in bundles if b.get("rows")]

    # -- overlay headline: wall time and throughput across GPUs, standalone --
    if with_scaling:
        overlay_sizes = sorted({r["nrow"] for b in with_scaling
                                for r in b["baseline"]})
        fig, axes = plt.subplots(1, 2, figsize=(3.4 * 2, 3.1))
        titles = ["Wall time", "Pixel throughput"]
        for mi, ((metric, label, scale, guide), title) in enumerate(zip(
                (METRIC_ROWS[0], METRIC_ROWS[2]), titles)):  # wall time, throughput
            ax = axes[mi]
            panel_max = None  # (y, x) of the highest plotted point
            for b in with_scaling:
                curve = b["baseline"]
                xs = [r["nrow"] for r in curve]
                ys = [r[metric] * scale for r in curve]
                ax.plot(xs, ys, marker="o", ms=3, lw=1.2, label=b["label"])
                for x, y in zip(xs, ys):
                    if y > 0 and (panel_max is None or y > panel_max[0]):
                        panel_max = (y, x)
            if panel_max is not None:
                plots.annotate_max(ax, panel_max[1], panel_max[0],
                                   overlay_sizes, plots.unit_of(label))
            if guide:
                anchor = max((b["baseline"][-1] for b in with_scaling),
                             key=lambda r: r["nrow"])
                ax.plot(overlay_sizes,
                        [anchor[metric] * scale * (x / anchor["nrow"]) ** 2
                         for x in overlay_sizes], ls="--", lw=1,
                        color="gray", label="$N^2$")
            ax.set_xscale("log")
            ax.set_yscale("log")
            plots.scaling_axes(ax, overlay_sizes)
            ax.set_ylabel(label)
            ax.set_xlabel("image side N [px]")
            ax.set_title(title)
        # one shared legend above both panels (the GPU colours and N^2 guide are
        # common to the pair), so each panel keeps its full plotting area.
        handles, labels_ = axes[0].get_legend_handles_labels()
        leg = fig.legend(handles, labels_, loc="outside upper center",
                         ncol=len(handles), fontsize=7)
        plots.save_fig(fig, outdir / "cross_gpu_scaling_overlay",
                       extra_artists=[leg])

    # -- per-GPU OFAT wall-time sweep, standalone grid --
    if sweep:
        all_ok = [r for b in sweep for r in b["rows"]]
        cols = [ax for ax in AXES
                if ax != "K" and len({r[ax] for r in all_ok}) > 1] or [AXES[0]]
        sweep_sizes = sorted({r["nrow"] for r in all_ok})
        metric, mlabel, mscale, mguide = METRIC_ROWS[0]  # wall time [s]

        fig, axes2d = plt.subplots(len(sweep), len(cols),
                                   figsize=(3.4 * len(cols), 2.7 * len(sweep)),
                                   squeeze=False, sharex=True)
        for bi, b in enumerate(sweep):
            ok = b["rows"]
            base = axis_baselines(ok)
            baseline = select(ok, base)
            for ci, ax_name in enumerate(cols):
                cell = axes2d[bi][ci]
                panel_max = None  # (y, x) of the highest plotted point
                for v in sorted({r[ax_name] for r in ok}):
                    fixed = {a: base[a] for a in AXES if a != ax_name}
                    fixed[ax_name] = v
                    rows_v = select(ok, fixed)
                    if not rows_v:
                        continue
                    xs = [r["nrow"] for r in rows_v]
                    ys = [r[metric] * mscale for r in rows_v]
                    cell.plot(xs, ys, marker="o", ms=3, lw=1.2,
                              label=f"{ax_name}={v}")
                    for x, y in zip(xs, ys):
                        if y > 0 and (panel_max is None or y > panel_max[0]):
                            panel_max = (y, x)
                if panel_max is not None:
                    plots.annotate_max(cell, panel_max[1], panel_max[0],
                                       sweep_sizes, plots.unit_of(mlabel))
                if mguide and len(baseline) >= 2:
                    x0, y0 = baseline[-1]["nrow"], baseline[-1][metric] * mscale
                    xg = sorted({r["nrow"] for r in ok})
                    cell.plot(xg, [y0 * (x / x0) ** 2 for x in xg],
                              ls="--", lw=1, color="gray", label="$N^2$")
                cell.set_xscale("log")
                cell.set_yscale("log")
                plots.scaling_axes(cell, sweep_sizes)
                if bi == 0:
                    cell.set_title(ax_name)
                    cell.legend()
                if bi == len(sweep) - 1:
                    cell.set_xlabel("image side N [px]")
                if ci == 0:
                    cell.set_ylabel(f"{b['label']}\n{mlabel}")
        plots.save_fig(fig, outdir / "cross_gpu_scaling")

    # ----- cross-GPU near-roof comparison (ncu) -----
    # The dominant kernels' DRAM throughput as % of each GPU's own peak: efficiency
    # holding across hardware generations is the portability argument. Kernels are
    # matched across bundles by ident and ranked by their best full-run share.
    ncu_bundles = [b for b in bundles if b["summary"].get("ncu", {}).get("kernels")]
    if ncu_bundles:
        # rank kernels (by ident, size variants already merged per bundle) by their
        # best full-run share across GPUs; the summaries keep every ident, so a
        # kernel that ranks differently per GPU is still found in each bundle.
        rank, lib_of = {}, {}
        for b in ncu_bundles:
            for k in b["summary"]["ncu"]["kernels"]:
                rank[k["ident"]] = max(rank.get(k["ident"], 0.0),
                                       k.get("full_run_pct") or 0.0)
                lib_of.setdefault(k["ident"], k.get("library", ""))
        # ascending so the highest-share kernel lands at the top of the barh
        chosen = [k for k, _ in sorted(rank.items(), key=lambda kv: kv[1])[-13:]]
        series = []
        for b in ncu_bundles:
            kd = {k["ident"]: k for k in b["summary"]["ncu"]["kernels"]}
            vals = [kd.get(key, {}).get("dram_pct", 0.0) for key in chosen]
            series.append((b["label"], vals))
        # Grouped (offset) bars: one clean sub-bar per GPU within each kernel row,
        # so no GPU is ever hidden behind or blended into another. Row pitch grows
        # with the GPU count so the group always fits; figure height follows.
        n = len(series)
        bar_h = 0.26
        pitch = bar_h * n + 0.26  # group height + inter-row gap
        y = np.arange(len(chosen)) * pitch
        fig, ax = plt.subplots(figsize=(7.5, 1.0 + 0.5 * len(chosen) * pitch))
        ax.set_axisbelow(True)  # grid behind the bars, not over them
        for i, (label, vals) in enumerate(series):
            offset = ((n - 1) / 2 - i) * bar_h  # first series on top of the group
            bars = ax.barh(y + offset, vals, height=bar_h, label=label)
            ax.bar_label(bars, fmt="%.0f", padding=2, fontsize=6)
        # tag third-party library kernels (cuFFT/cuBLAS/CUB, flagged at the ncu
        # stage) and grey their labels so they read as not-ours, bars unchanged.
        tags = [lib_of.get(c, "") for c in chosen]
        labels = [c + (f"  [{t}]" if t else "") for c, t in zip(chosen, tags)]
        ax.set_yticks(y, labels)
        for tick, t in zip(ax.get_yticklabels(), tags):
            if t:
                tick.set_color("0.6")
        ax.set_xlabel("DRAM throughput [% of peak]")
        ax.set_xlim(0, 105)
        ax.set_title("Dominant-kernel DRAM efficiency across GPUs")
        ax.legend(fontsize=7, loc="lower right")
        plots.save_fig(fig, outdir / "cross_gpu_ncu_efficiency")

    # ----- cross-GPU FP32 roofline overlay -----
    # Every profiled GPU's dominant kernels on one roofline. Colour encodes the host
    # GPU (its roof, drawn in the same colour); bubble area is the kernel's share of
    # GPU time. Every bubble is numbered and a side key maps the number to the kernel
    # name -- numbered by kernel identity, so the same kernel carries the same number
    # on every card and its three coloured points read as one kernel across GPUs. The
    # time-dominant kernels clustering on the bandwidth slope, GPU after GPU, is the
    # cross-hardware "bandwidth-bound everywhere" argument in one figure.
    def _has_roof(b):
        ncu = b["summary"].get("ncu", {})
        return (ncu.get("peak_dram_gbps") and ncu.get("peak_fp32_gflops")
                and any(k.get("ai") and k.get("gflops") for k in ncu.get("kernels", [])))

    roof_bundles = [b for b in bundles if _has_roof(b)]
    if roof_bundles:
        import matplotlib.patheffects as pe
        from matplotlib.lines import Line2D

        cmap = plt.get_cmap("tab10")

        def kshare(k):
            return k.get("full_run_pct") or 0.0

        # number kernels by identity (shared across GPUs); rank by best full-run share
        # and keep the top 15 -- the long tail just crowds the figure and the key.
        rank, lib_of = {}, {}
        for b in roof_bundles:
            for k in b["summary"]["ncu"]["kernels"]:
                if k.get("ai") and k.get("gflops"):
                    rank[k["ident"]] = max(rank.get(k["ident"], 0.0), kshare(k))
                    lib_of.setdefault(k["ident"], k.get("library", ""))
        chosen = [c for c, _ in sorted(rank.items(), key=lambda kv: kv[1],
                                       reverse=True)[:15]]
        num = {c: i + 1 for i, c in enumerate(chosen)}

        pts = [(k["ai"], k["gflops"]) for b in roof_bundles
               for k in b["summary"]["ncu"]["kernels"]
               if k["ident"] in num and k.get("ai") and k.get("gflops")]
        ridges = [b["summary"]["ncu"]["peak_fp32_gflops"]
                  / b["summary"]["ncu"]["peak_dram_gbps"] for b in roof_bundles]
        xlo = min(min(a for a, _ in pts), min(ridges)) / 3
        xhi = max(max(a for a, _ in pts), max(ridges)) * 3
        xs = np.geomspace(xlo, xhi, 200)

        fig, ax = plt.subplots(figsize=(6.8, 5.2), constrained_layout=False)
        gpu_handles = []
        for i, b in enumerate(roof_bundles):
            peak_gbps = b["summary"]["ncu"]["peak_dram_gbps"]
            peak_fp32 = b["summary"]["ncu"]["peak_fp32_gflops"]
            color = cmap(i % 10)
            # GB/s * FLOP/byte = GFLOP/s; flat compute roof beyond the ridge.
            ax.plot(xs, np.minimum(peak_gbps * xs, peak_fp32), color=color, lw=1.2,
                    zorder=1)
            for k in b["summary"]["ncu"]["kernels"]:
                if k["ident"] not in num or not (k.get("ai") and k.get("gflops")):
                    continue
                ax.scatter(k["ai"], k["gflops"], s=20 + 7 * kshare(k), color=color,
                           marker="o", alpha=0.7, edgecolor="k", linewidth=0.4,
                           zorder=3)
                ax.annotate(str(num[k["ident"]]), (k["ai"], k["gflops"]), fontsize=5,
                            fontweight="bold", ha="center", va="center", zorder=5,
                            color="black", path_effects=[
                                pe.withStroke(linewidth=1.2, foreground="white")])
            gpu_handles.append(Line2D([], [], color=color, marker="o", ls="-",
                                      markeredgecolor="k", markeredgewidth=0.4,
                                      label=b["label"]))
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("arithmetic intensity [FLOP / byte]")
        ax.set_ylabel("performance [GFLOP/s]")
        ax.set_title("FP32 roofline across GPUs (bubble area = % of GPU time)")
        # both legends inside the upper-left empty triangle (above the roofs): the GPU
        # colour key on top, the number -> kernel name key stacked just under it.
        kh = [Line2D([], [], marker="o", ls="", ms=4, markerfacecolor="0.7",
                     markeredgecolor="k", markeredgewidth=0.4,
                     label=f"{num[c]}. {c}" + (f"  [{lib_of[c]}]" if lib_of[c] else ""))
              for c in chosen]
        kleg = ax.legend(handles=kh, loc="upper left", bbox_to_anchor=(0.01, 0.83),
                         borderaxespad=0.0, fontsize=5, labelspacing=0.25,
                         handletextpad=0.3, title="kernel", title_fontsize=6,
                         framealpha=0.9)
        ax.add_artist(kleg)
        # bubble-size key: representative shares mapped through s = 20 + 7*share, with
        # marker size = sqrt(s) since scatter `s` is area (points^2).
        sh = [Line2D([], [], marker="o", ls="", markerfacecolor="0.7",
                     markeredgecolor="k", markeredgewidth=0.4,
                     markersize=np.sqrt(20 + 7 * v), label=f"{v}%")
              for v in (1, 10, 25, 50)]
        # handleheight just clears the largest marker (50%, diameter sqrt(20+7*50)
        # ~ 19 pt ~ 3.2x the 6 pt font); everything else kept tight so the key
        # doesn't grow tall.
        sleg = ax.legend(handles=sh, loc="lower right", fontsize=6, labelspacing=0.3,
                         handletextpad=0.6, borderpad=0.5, handleheight=3.3,
                         handlelength=1.8, title="% of GPU time", title_fontsize=6,
                         framealpha=0.9)
        ax.add_artist(sleg)
        ax.legend(handles=gpu_handles, loc="upper left", bbox_to_anchor=(0.01, 0.99),
                  fontsize=7, title="GPU", title_fontsize=7, framealpha=0.9)
        plots.save_fig(fig, outdir / "cross_gpu_roofline")

    # ----- cross-GPU per-major-cycle runtime -----
    # Same real-data deconvolution on each card: the per-cycle minor-loop wall time
    # grouped by major cycle, one bar per GPU. The iteration count per cycle is fixed
    # by the data (identical across cards to <1%, annotated once), so the spread is
    # pure runtime and tracks memory bandwidth cycle after cycle.
    def _cycles(b):
        p = b["path"] / "realdata" / "cycles.csv"
        return ([{k: to_num(v) for k, v in r.items()} for r in read_csv_rows(p)]
                if p.exists() else [])

    rt = [(b, _cycles(b)) for b in bundles]
    rt = [(b, c) for b, c in rt if c]
    # longest-running GPU first, so each group's bars descend left-to-right
    # (and the legend follows the same order).
    rt.sort(key=lambda bc: sum(r["elapsed_ms"] for r in bc[1]), reverse=True)
    if rt:
        cmap = plt.get_cmap("tab10")
        cycle_ids = sorted({r["cycle_id"] for _, c in rt for r in c})
        x = np.arange(len(cycle_ids))
        bw = 0.8 / len(rt)
        ymax = max(r["elapsed_ms"] / 1e3 for _, c in rt for r in c)
        fig, ax = plt.subplots(figsize=(7.0, 4.0))
        ax.set_axisbelow(True)  # grid behind the bars
        # colour keyed to the bundle's original position so each GPU keeps the
        # same colour as in the other cross-GPU figures, independent of bar order.
        color_of = {id(b): i for i, b in enumerate(bundles)}
        for i, (b, c) in enumerate(rt):
            by_cycle = {r["cycle_id"]: r["elapsed_ms"] / 1e3 for r in c}
            vals = [by_cycle.get(cid, 0.0) for cid in cycle_ids]
            ax.bar(x + (i - (len(rt) - 1) / 2) * bw, vals, width=bw,
                   color=cmap(color_of[id(b)] % 10), label=b["label"])
        # iteration count per cycle (work is data-fixed, ~equal across cards): annotate
        # once above each group from the first bundle that has the cycle.
        iters = {}
        for _, c in rt:
            for r in c:
                iters.setdefault(r["cycle_id"], int(r["total_iterations"]))
        for xi, cid in zip(x, cycle_ids):
            ax.text(xi, ymax * 1.02, f"{iters[cid]:,} it", ha="center", va="bottom",
                    fontsize=6, color="0.35")
        # the final cycle runs auto-masking (a full-image dilation/RMS/threshold pass),
        # so it costs far more than its low iteration count alone would predict.
        mask_top = max(r["elapsed_ms"] / 1e3 for _, c in rt for r in c
                       if r["cycle_id"] == cycle_ids[-1])
        ax.text(x[-1], mask_top + ymax * 0.02, "Auto-masking", ha="center",
                va="bottom", fontsize=7, color="0.3")
        ax.set_xticks(x, [str(c) for c in cycle_ids])
        ax.set_ylim(top=ymax * 1.13)
        ax.set_xlabel("major cycle")
        ax.set_ylabel("minor-cycle wall time [s]")
        ax.set_title("Per-major-cycle runtime across GPUs")
        ax.legend(fontsize=7)
        plots.save_fig(fig, outdir / "cross_gpu_cycle_timing")

    # ----- hardware table -----
    rows = []
    for b in bundles:
        gpu = b["env"].get("gpu", {})
        git = b["env"].get("git", {})
        rows.append([b["label"], gpu.get("name", "?"), gpu.get("memory.total", "?"),
                     gpu.get("driver_version", "?"),
                     b["env"].get("cuda_toolkit", "?"),
                     git.get("commit", "?")[:10]])
    tables.write_table(
        ["label", "GPU", "memory [MiB]", "driver", "CUDA", "commit"],
        rows, outdir / "hardware", title="Hardware used")

    # ----- throughput / efficiency table -----
    rows = []
    for b in bundles:
        s = b["summary"]
        sc = s.get("scaling", {})
        at = sc.get("at_largest_baseline_size", {})
        ncu = s.get("ncu", {})
        rows.append([
            b["label"],
            at.get("nrow", "-"),
            at.get("wall_s", "-"),
            at.get("ms_per_iter", "-"),
            at.get("mpix_iter_per_s", "-"),
            at.get("used_mem_mb", "-"),
            _fmt(sc.get("exponent_time_vs_side")),
            _fmt(ncu.get("time_weighted_dram_pct")),
            _fmt(ncu.get("top10_full_run_pct")),
        ])
    tables.write_table(
        ["label", "size N", "wall [s]", "ms/iter", "Mpix·iter/s", "mem [MiB]",
         "time ~ N^p", "DRAM % of peak", "top-10 kernels [% GPU time]"],
        rows, outdir / "throughput", title="Throughput and efficiency summary")

    # ----- tiled argmax table -----
    rows = []
    for b in bundles:
        am = b["summary"].get("argmax", {})
        native_ms = am.get("native_ms")
        for psf, best in am.get("per_footprint", {}).items():
            rows.append([b["label"], psf, best.get("best_tile"),
                         native_ms, best.get("full_ms"), best.get("incr_ms"),
                         best.get("speedup_incr"), best.get("breakeven_n")])
    if rows:
        tables.write_table(
            ["label", "footprint [px]", "best tile [px]", "native [ms]",
             "reseed [ms]", "incremental [ms]", "speedup vs native",
             "breakeven N"],
            rows, outdir / "argmax", title="Incremental argmax: best configurations")

    print(f"Aggregated {len(bundles)} bundle(s) -> {outdir}")
    return {"bundles": [str(b['path']) for b in bundles], "outdir": str(outdir)}


def _fmt(v):
    return f"{v:.2f}" if isinstance(v, (int, float)) else "-"
