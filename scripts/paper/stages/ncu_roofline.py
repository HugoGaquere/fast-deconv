"""Stage 4 -- hardware efficiency of the dominant kernels via Nsight Compute.

Profiles a short iteration-bounded run of the first real-data cycle
(example_wscms on a FastDDFacet dump, capped to one scale selection x a few clean
iters) with an explicit metric list (much faster than --set full), aggregates per
kernel, and reports achieved DRAM bandwidth and %-of-peak DRAM/SM throughput. A
memory-bound kernel near the DRAM roof is the no-baseline-needed argument that the
implementation is efficient; per-kernel efficiency is steady across iterations, so
the bounded run is representative of the full cycle.

When the bundle also has the nsys stage's per-kernel time summary (the same
first real-data cycle), every kernel is cross-referenced with its share of GPU
time in that run: the table and figure are ranked by full-run share and report
the cumulative share the displayed kernels cover -- the evidence that the
kernels whose efficiency we show are the ones that dominate the runtime.

ncu needs GPU performance-counter permission; on failure the stage reports the
fix (run as root or set kernel module option NVreg_RestrictProfilingToAdminUsers=0).
"""
from __future__ import annotations

import csv
import io
import re
import shutil
from pathlib import Path

from harness import (Ctx, SkipStage, capture_cmd, first_cycle, iteration_flags,
                     kernel_ident, read_csv_rows, run_cmd, to_num, write_json)

M_TIME = "gpu__time_duration.sum"
M_DRAM_PCT = "dram__throughput.avg.pct_of_peak_sustained_elapsed"
M_SM_PCT = "sm__throughput.avg.pct_of_peak_sustained_elapsed"
M_BYTES = "dram__bytes.sum"
# FP32 FLOP instruction counts for the roofline: performance = fadd + fmul + 2*ffma.
M_FADD = "sm__sass_thread_inst_executed_op_fadd_pred_on.sum"
M_FMUL = "sm__sass_thread_inst_executed_op_fmul_pred_on.sum"
M_FFMA = "sm__sass_thread_inst_executed_op_ffma_pred_on.sum"
# Achieved (measured) occupancy -- the real counterpart to the free theoretical
# occupancy -- and L2 sector hit rate (cache reuse for the memory-bound kernels).
M_OCC_ACH = "sm__warps_active.avg.pct_of_peak_sustained_active"
M_L2_HIT = "lts__t_sector_hit_rate.pct"
METRICS = [M_TIME, M_DRAM_PCT, M_SM_PCT, M_BYTES,
           M_FADD, M_FMUL, M_FFMA, M_OCC_ACH, M_L2_HIT]

# Newer ncu versions abbreviate the unit names (ms, not msecond).
_TIME_SCALE = {"nsecond": 1e-9, "usecond": 1e-6, "msecond": 1e-3, "second": 1.0,
               "ns": 1e-9, "us": 1e-6, "ms": 1e-3, "s": 1.0}
_BYTE_SCALE = {"byte": 1.0, "Kbyte": 1e3, "Mbyte": 1e6, "Gbyte": 1e9}

# Columns ncu always emits alongside the requested metrics, at no extra pass:
# theoretical occupancy (derived from the launch config, not measured), the
# resource that caps it, and the device specs we turn into the DRAM roof.
M_OCC = "sm__maximum_warps_per_active_cycle_pct"
M_REGS = "launch__registers_per_thread"
M_SMEM = "launch__shared_mem_per_block"
_OCC_LIMITS = {"registers": "launch__occupancy_limit_registers",
               "shared mem": "launch__occupancy_limit_shared_mem",
               "warps": "launch__occupancy_limit_warps",
               "blocks": "launch__occupancy_limit_blocks",
               "barriers": "launch__occupancy_limit_barriers"}
_A_BUSWIDTH = "device__attribute_global_memory_bus_width"   # bits
_A_MEMCLK = "device__attribute_max_mem_frequency_khz"       # kHz


def _peak_dram_gbps(recs: list[dict]) -> float:
    """Hardware DRAM bandwidth roof from the device attributes ncu records:
    bus_width_bits/8 * mem_clock_hz * 2 (double data rate). 0 when absent."""
    for r in recs:
        bw, mc = to_num(r.get(_A_BUSWIDTH, "")), to_num(r.get(_A_MEMCLK, ""))
        if isinstance(bw, (int, float)) and isinstance(mc, (int, float)) and bw and mc:
            return bw / 8 * mc * 1e3 * 2 / 1e9
    return 0.0


# FP32 CUDA cores per SM by compute capability (not exposed as a device attr).
_FP32_CORES_PER_SM = {(7, 0): 64, (7, 5): 64, (8, 0): 64, (8, 6): 128,
                      (8, 9): 128, (9, 0): 128}


def _peak_fp32_gflops(recs: list[dict]) -> float:
    """Peak FP32 throughput (GFLOP/s) for the compute roof: SMs * cores/SM * 2
    (FMA) * clock, with cores/SM looked up from the compute capability. 0 if the
    capability is unknown or the attributes are absent."""
    for r in recs:
        cc = (to_num(r.get("device__attribute_compute_capability_major", "")),
              to_num(r.get("device__attribute_compute_capability_minor", "")))
        sms = to_num(r.get("device__attribute_multiprocessor_count", ""))
        clk = to_num(r.get("device__attribute_max_gpu_frequency_khz", ""))
        cores = _FP32_CORES_PER_SM.get(cc)
        if cores and isinstance(sms, (int, float)) and isinstance(clk, (int, float)) \
                and sms and clk:
            return sms * cores * 2 * (clk * 1e3) / 1e9
    return 0.0


def _library(name: str, phase: str) -> str:
    """Third-party library a kernel belongs to, for tagging it as not-ours.
    cuFFT/cuBLAS are reliable from the phase; CUB shares the reductions phase with
    our own kernels, so it is detected from the cub:: namespace in the name."""
    low = name.lower()
    if "cuFFT" in phase:
        return "cuFFT"
    if "cuBLAS" in phase or "cublas" in low:
        return "cuBLAS"
    if "cub::" in low or "cub_" in low:
        return "CUB"
    return ""


def _occupancy_limiter(rec: dict) -> str:
    """Resource allowing the fewest resident blocks/SM -- what caps theoretical
    occupancy for this launch (ties, the fully-occupied case, are joined)."""
    vals = {n: to_num(rec.get(c, "")) for n, c in _OCC_LIMITS.items()}
    vals = {n: v for n, v in vals.items() if isinstance(v, (int, float)) and v > 0}
    if not vals:
        return "-"
    lo = min(vals.values())
    return "+".join(n for n, v in vals.items() if v == lo)


def _full_run_shares(nsys_csv: Path) -> dict[str, float]:
    """Identifier -> % of total GPU time in the nsys stage's representative run."""
    if not nsys_csv.exists():
        return {}
    rows = read_csv_rows(nsys_csv)
    if not rows:
        return {}
    name_col = next(c for c in rows[0] if c.strip() == "Name")
    time_col = next(c for c in rows[0] if c.strip().startswith("Total Time"))
    acc: dict[str, float] = {}
    for r in rows:
        t = to_num(r[time_col])
        if isinstance(t, (int, float)):
            key = kernel_ident(r[name_col])
            acc[key] = acc.get(key, 0.0) + float(t)
    total = sum(acc.values())
    if total <= 0:
        return {}
    return {k: 100.0 * v / total for k, v in acc.items()}


def _dims_product(grid: str) -> int:
    """Product of the grid dimensions from an ncu "(x, y, z)" cell -- one number
    standing in for the launch size, used to tell apart same-named kernels
    launched at different problem sizes (1 when the column is absent)."""
    p = 1
    for n in re.findall(r"\d+", grid or ""):
        p *= int(n)
    return p


def run(ctx: Ctx) -> dict:
    if shutil.which("ncu") is None:
        raise SkipStage("ncu not found in PATH -- install Nsight Compute to collect "
                        "roofline data")
    if not ctx.dump_dir:
        raise SkipStage("no --dump-dir given -- the roofline profiles the first "
                        "real-data cycle (example_wscms on a FastDDFacet dump)")
    cfg = ctx.preset["ncu"]
    out = ctx.stage_dir("ncu")
    rep = out / "wscms"

    # Profile every launch of the iteration-bounded run (one scale selection x a
    # few clean iters) so the clean-loop kernels are covered, not just the opening
    # cuFFT storm: scale selection alone issues ~n_scales*n_facet*5 launches, so
    # any --launch-count cap truncates before the clean loop. Application replay
    # makes profiling the whole run affordable -- it re-runs the bounded app once
    # per metric pass instead of replaying each kernel in place, so cost is a fixed
    # handful of app runs regardless of launch count (and no per-kernel device
    # memory save/restore). The bound's fixed iteration count keeps the per-run
    # kernel sequence stable across passes, which application replay requires.
    # max_launches stays an optional runaway guard for an unbounded cfg.
    launch_cap = (["--launch-count", str(cfg["max_launches"])]
                  if cfg.get("max_launches") else [])
    rc = run_cmd(ctx, [
        "ncu", "-f", f"--export={rep}",
        "--replay-mode", "application",
        f"--metrics={','.join(METRICS)}",
        *launch_cap,
        ctx.binary("example_wscms"),
        ctx.dump_dir,
        f"--cycles={first_cycle(ctx.cycles)}",
        f"--device={ctx.device}",
        *iteration_flags(cfg),
    ], "ncu_profile", check=False)
    if rc != 0 or not rep.with_suffix(".ncu-rep").exists():
        log_tail = (ctx.bundle / "logs" / "ncu_profile.log").read_text()[-2000:]
        if "ERR_NVGPUCTRPERM" in log_tail:
            raise RuntimeError(
                "ncu lacks GPU performance-counter permission. Run the harness as "
                "root, or set the nvidia kernel module option "
                "NVreg_RestrictProfilingToAdminUsers=0 and reboot.")
        raise RuntimeError("ncu profiling failed (see logs/ncu_profile.log)")

    raw = capture_cmd(["ncu", "--import", rep.with_suffix(".ncu-rep"),
                       "--csv", "--page", "raw"])
    (out / "raw.csv").write_text(raw)
    summary = analyze(raw, out, ctx.bundle / "nsys" / "stats_cuda_gpu_kern_sum.csv")
    write_json(out / "summary.json", summary)
    return summary


def _parse_raw(raw: str) -> tuple[list[dict], dict]:
    rows = list(csv.reader(io.StringIO(raw)))
    hi = next(i for i, r in enumerate(rows) if "Kernel Name" in r)
    header = rows[hi]
    body = hi + 1
    units: dict = {}
    # ncu emits a units row right after the header (its ID field is not numeric).
    id_idx = header.index("ID") if "ID" in header else 0
    if body < len(rows) and not str(rows[body][id_idx]).strip().isdigit():
        units = dict(zip(header, rows[body]))
        body += 1
    recs = [dict(zip(header, r)) for r in rows[body:] if len(r) == len(header)]
    return recs, units


def analyze(raw: str, out, nsys_csv=None) -> dict:
    import matplotlib.pyplot as plt
    import numpy as np

    from analysis import plots, tables
    from stages.nsys_breakdown import kernel_phase
    plots.setup_style()

    recs, units = _parse_raw(raw)
    peak_gbps = _peak_dram_gbps(recs)
    peak_fp32 = _peak_fp32_gflops(recs)
    # Newer metrics only present after a re-profile; guard so old bundles still work.
    have_flops = bool(recs) and M_FFMA in recs[0]
    have_occ_ach = bool(recs) and M_OCC_ACH in recs[0]
    have_l2 = bool(recs) and M_L2_HIT in recs[0]
    t_scale = _TIME_SCALE.get(units.get(M_TIME, "nsecond"), 1e-9)
    b_scale = _BYTE_SCALE.get(units.get(M_BYTES, "byte"), 1.0)
    shares = _full_run_shares(Path(nsys_csv)) if nsys_csv else {}

    def num(rec, col):
        v = to_num(rec.get(col, 0))
        return float(v) if isinstance(v, (int, float)) else 0.0

    def short(n, w=60):
        head = n.split("<")[0].split("(")[0].removeprefix("void ").strip()
        return (head or n)[:w]

    def libtag(k):
        return f"  [{k['library']}]" if k.get("library") else ""

    def grey_libs(ax, ordered):
        # grey the y-tick labels of third-party (cuFFT/cuBLAS/CUB) kernels so they
        # read as not-ours; `ordered` must match the tick order top-to-bottom.
        for tick, k in zip(ax.get_yticklabels(), ordered):
            if k.get("library"):
                tick.set_color("0.6")

    # Group by (base name, launch size), not base name alone: one demangled name
    # like gemv2N_kernel covers both the precompute gains gemv and the
    # spectral-fit gemv at different problem sizes, and merging them would blend
    # their efficiencies. Launch size is the product of the grid dimensions.
    per: dict[tuple, dict] = {}
    for r in recs:
        name = r["Kernel Name"]
        ident = kernel_ident(name)
        size = _dims_product(r.get("Grid Size", ""))
        t = num(r, M_TIME) * t_scale
        d = per.setdefault((ident, size),
                           {"name": name, "ident": ident, "size": size,
                            "phase": kernel_phase(name),
                            "launches": 0, "time_s": 0.0, "bytes": 0.0,
                            "dram_w": 0.0, "sm_w": 0.0, "occ_w": 0.0,
                            "flops": 0.0, "occ_ach_w": 0.0, "l2_w": 0.0,
                            "regs": 0.0, "smem": 0.0, "limiter": "-"})
        d["launches"] += 1
        d["time_s"] += t
        d["bytes"] += num(r, M_BYTES) * b_scale
        d["dram_w"] += num(r, M_DRAM_PCT) * t
        d["sm_w"] += num(r, M_SM_PCT) * t
        d["occ_w"] += num(r, M_OCC) * t
        d["flops"] += num(r, M_FADD) + num(r, M_FMUL) + 2 * num(r, M_FFMA)
        d["occ_ach_w"] += num(r, M_OCC_ACH) * t
        d["l2_w"] += num(r, M_L2_HIT) * t
        d["regs"] = num(r, M_REGS) or d["regs"]
        d["smem"] = num(r, M_SMEM) or d["smem"]
        d["limiter"] = _occupancy_limiter(r)

    total_s = sum(d["time_s"] for d in per.values()) or 1e-12
    # nsys reports full-run time only by base name, so split a base name's share
    # across its size variants in proportion to their profiled time, and tag the
    # label with the grid size when a name spans more than one size.
    ident_time: dict[str, float] = {}
    ident_sizes: dict[str, set] = {}
    for d in per.values():
        ident_time[d["ident"]] = ident_time.get(d["ident"], 0.0) + d["time_s"]
        ident_sizes.setdefault(d["ident"], set()).add(d["size"])

    kernels = []
    for d in per.values():
        if d["time_s"] <= 0:
            continue
        base_share = shares.get(d["ident"]) if shares else None
        multi = len(ident_sizes[d["ident"]]) > 1
        kernels.append({
            "name": d["name"],
            "ident": d["ident"],
            "phase": d["phase"],
            "library": _library(d["name"], d["phase"]),
            "label": short(d["name"]) + (f"  [grid {d['size']:,}]" if multi else ""),
            "launches": d["launches"],
            "time_ms": d["time_s"] * 1e3,
            "profiled_pct": 100 * d["time_s"] / total_s,
            "full_run_pct": (base_share * d["time_s"] / ident_time[d["ident"]]
                             if base_share is not None and ident_time[d["ident"]] > 0
                             else (base_share if shares else None)),
            "gbps": d["bytes"] / 1e9 / d["time_s"],
            "bytes_gb": d["bytes"] / 1e9,
            "dram_pct": d["dram_w"] / d["time_s"],
            "sm_pct": d["sm_w"] / d["time_s"],
            "occupancy_pct": d["occ_w"] / d["time_s"],
            # roofline coordinates: achieved GFLOP/s and arithmetic intensity
            # (FLOP per DRAM byte). Only meaningful when the FLOP metrics are in.
            "gflops": d["flops"] / 1e9 / d["time_s"],
            "ai": (d["flops"] / d["bytes"]) if d["bytes"] > 0 else 0.0,
            "occupancy_ach_pct": d["occ_ach_w"] / d["time_s"],
            "l2_hit_pct": d["l2_w"] / d["time_s"],
            "regs": d["regs"],
            "smem": d["smem"],
            "limiter": d["limiter"],
        })
    # Rank by share of the representative nsys run when its data is in the
    # bundle -- the paper's "these kernels dominate the runtime" ordering --
    # otherwise by time within the profiled run.
    if shares:
        kernels.sort(key=lambda k: k["full_run_pct"], reverse=True)
    else:
        kernels.sort(key=lambda k: k["time_ms"], reverse=True)
    top = kernels[:10]
    coverage = sum(k["full_run_pct"] for k in top) if shares else None

    title = "Kernel bandwidth utilization (Nsight Compute)"
    if coverage is not None:
        title += (f"; these {len(top)} kernels are {coverage:.0f}% of full-run "
                  "GPU time (Nsight Systems)")
    tables.write_table(
        ["kernel", "launches", "profiled_ms", "profiled_pct", "full_run_pct",
         "achieved_GBps", "dram_pct_of_peak", "sm_pct_of_peak"],
        [[k["label"] + libtag(k), k["launches"], f"{k['time_ms']:.3f}",
          f"{k['profiled_pct']:.1f}",
          "-" if k["full_run_pct"] is None else f"{k['full_run_pct']:.1f}",
          f"{k['gbps']:.0f}", f"{k['dram_pct']:.1f}",
          f"{k['sm_pct']:.1f}"] for k in top],
        out / "kernels", title=title)

    pitch = 0.7  # row spacing < 1 packs the kernel groups closer together
    fig, ax = plt.subplots(figsize=(6.0, 0.8 + 0.5 * len(top) * pitch))
    ax.set_axisbelow(True)  # grid behind the bars
    names = [k["label"] + libtag(k) for k in top]
    if coverage is not None:
        names = [f"{n}  ({k['full_run_pct']:.1f}%)" for n, k in zip(names, top)]
    names = names[::-1]
    y = np.arange(len(top)) * pitch
    ax.barh(y + 0.16, [k["dram_pct"] for k in top][::-1], height=0.30,
            label="DRAM % of peak", color="#3b738f")
    ax.barh(y - 0.16, [k["sm_pct"] for k in top][::-1], height=0.30,
            label="SM % of peak", color="#d1605e")
    ax.set_yticks(y, names)
    grey_libs(ax, top[::-1])
    ax.set_xlabel("% of peak sustained throughput")
    ax.set_xlim(0, 100)
    ax.set_title("Dominant kernels vs hardware limits"
                 + (f" -- {coverage:.0f}% of GPU time"
                    if coverage is not None else ""))
    ax.legend(loc="lower right")
    plots.save_fig(fig, out / "kernel_utilization")

    # share of GPU time used to size/rank kernels: full-run (nsys cross-ref) when
    # available, else the share within the profiled run.
    def share(k):
        return k["full_run_pct"] if k["full_run_pct"] is not None else k["profiled_pct"]

    # --- roofline-regime scatter: place every kernel by how close it runs to the
    # DRAM and SM roofs, sized by its share of GPU time. The time-dominant kernels
    # sitting high on the DRAM axis is the memory-bound, near-roof efficiency
    # argument in one figure. (Throughput regimes, not a FLOP/byte roofline.)
    cmap = plt.get_cmap("tab10")
    phase_list = sorted({k["phase"] for k in kernels})
    pcolor = {p: cmap(i % 10) for i, p in enumerate(phase_list)}
    import matplotlib.patheffects as pe
    from matplotlib.lines import Line2D

    def identify(ax, ranked, xs, ys, phase_loc="lower left"):
        """Number every bubble and list number -> kernel name in a side legend, so
        each point is identifiable (only the few largest used to be labelled). A
        second small legend keeps the phase colour key."""
        for i, (k, x, y) in enumerate(zip(ranked, xs, ys), 1):
            ax.annotate(str(i), (x, y), fontsize=5, fontweight="bold", ha="center",
                        va="center", zorder=5, color="black",
                        path_effects=[pe.withStroke(linewidth=1.2, foreground="white")])
        kh = [Line2D([], [], marker="o", ls="", ms=5, markeredgecolor="k",
                     markeredgewidth=0.4, markerfacecolor=pcolor[k["phase"]],
                     label=f"{i}. {short(k['name'], 28)}{libtag(k)}")
              for i, k in enumerate(ranked, 1)]
        kleg = ax.legend(handles=kh, loc="upper left", bbox_to_anchor=(1.02, 1.0),
                         borderaxespad=0.0, fontsize=5, labelspacing=0.25,
                         handletextpad=0.4,
                         title="kernel (bubble area = % of GPU time)",
                         title_fontsize=6)
        ax.add_artist(kleg)
        ph = [Line2D([], [], marker="o", ls="", ms=6, markeredgecolor="k",
                     markeredgewidth=0.4, markerfacecolor=pcolor[p], label=p)
              for p in phase_list]
        ax.legend(handles=ph, loc=phase_loc, fontsize=6, title="phase",
                  title_fontsize=6, framealpha=0.9)
        return kleg  # out-of-axes legend; pass to save_fig so it isn't clipped

    # constrained_layout off: it clips the kernel legend anchored outside the axes.
    # The legend is handed to save_fig as a bbox-extra artist so the tight save
    # grows the canvas to include it.
    # Only the 15 kernels with the largest share of GPU time -- the long tail of
    # tiny kernels just crowds the figure and the number->name legend.
    ranked = sorted(kernels, key=share, reverse=True)[:15]
    fig, ax = plt.subplots(figsize=(6.4, 5.2), constrained_layout=False)
    for p in phase_list:
        ks = [k for k in ranked if k["phase"] == p]
        ax.scatter([k["sm_pct"] for k in ks], [k["dram_pct"] for k in ks],
                   s=[20 + 7 * share(k) for k in ks], color=pcolor[p],
                   alpha=0.6, edgecolor="k", linewidth=0.4)
    ax.axhline(100, ls="--", lw=1, color="#3b738f")
    ax.axvline(100, ls="--", lw=1, color="#d1605e")
    ax.text(2, 99, "DRAM roof", fontsize=7, color="#3b738f", va="top")
    ax.text(99, 2, "SM roof", fontsize=7, color="#d1605e", ha="right")
    kleg = identify(ax, ranked, [k["sm_pct"] for k in ranked],
                    [k["dram_pct"] for k in ranked])
    ax.set_xlabel("SM throughput [% of peak]")
    ax.set_ylabel("DRAM throughput [% of peak]")
    ax.set_xlim(0, 105)
    ax.set_ylim(0, 105)
    ax.set_title("Kernel roofline regime (bubble area = % of GPU time)")
    plots.save_fig(fig, out / "roofline_regime", extra_artists=[kleg])

    # --- classic roofline: achieved GFLOP/s vs arithmetic intensity (FLOP/byte),
    # with the diagonal DRAM-bandwidth roof rising to the ridge point and the flat
    # FP32 compute roof beyond it. Needs the FLOP metrics and both peaks.
    rk = [k for k in kernels if have_flops and k["gflops"] > 0 and k["ai"] > 0]
    if rk and peak_gbps > 0 and peak_fp32 > 0:
        peak_bw_bps = peak_gbps * 1e9            # bytes/s for the diagonal slope
        ridge = peak_fp32 * 1e9 / peak_bw_bps    # AI where the two roofs meet
        ais = [k["ai"] for k in rk]
        xlo = min(min(ais), ridge) / 3
        xhi = max(max(ais), ridge) * 3
        xs = np.geomspace(xlo, xhi, 200)
        roof = np.minimum(peak_bw_bps * xs / 1e9, peak_fp32)  # GFLOP/s
        fig, ax = plt.subplots(figsize=(6.6, 5.0), constrained_layout=False)
        ax.plot(xs, roof, color="black", lw=1.5, zorder=1)
        ax.axvline(ridge, ls=":", lw=0.8, color="0.5")
        ax.text(ridge, peak_fp32 * 1.05, f"ridge {ridge:.1f}", fontsize=6,
                color="0.4", ha="center")
        ax.text(xhi, peak_fp32 * 1.05, f"FP32 {peak_fp32 / 1e3:.1f} TFLOP/s",
                fontsize=7, ha="right", va="bottom")
        ax.text(xlo * 1.1, peak_bw_bps * xlo * 1.1 / 1e9,
                f"{peak_gbps / 1e3:.2f} TB/s", fontsize=7, rotation=37,
                rotation_mode="anchor", va="bottom")
        ranked_rk = sorted(rk, key=share, reverse=True)[:15]  # top GPU-time share
        for k in ranked_rk:
            ax.scatter(k["ai"], k["gflops"], s=20 + 7 * share(k),
                       color=pcolor[k["phase"]], alpha=0.65, edgecolor="k",
                       linewidth=0.4, zorder=3)
        rkleg = identify(ax, ranked_rk, [k["ai"] for k in ranked_rk],
                         [k["gflops"] for k in ranked_rk], phase_loc="lower right")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("arithmetic intensity [FLOP / byte]")
        ax.set_ylabel("performance [GFLOP/s]")
        ax.set_title("FP32 roofline (bubble area = % of GPU time)")
        plots.save_fig(fig, out / "roofline", extra_artists=[rkleg])

    # --- DRAM traffic by phase: the memory-movement companion to the nsys
    # time-by-phase breakdown. FFT dominating bytes moved as well as time is why
    # DRAM is the right roof to measure against.
    by_phase = {}
    for k in kernels:
        by_phase[k["phase"]] = by_phase.get(k["phase"], 0.0) + k["bytes_gb"]
    tot_b = sum(by_phase.values()) or 1e-12
    phb = sorted(by_phase.items(), key=lambda kv: kv[1], reverse=True)
    tables.write_table(
        ["phase", "dram_GB", "share_pct"],
        [[p, f"{b:.2f}", f"{100 * b / tot_b:.1f}"] for p, b in phb],
        out / "dram_traffic_by_phase", title="DRAM traffic by algorithm phase")
    pitch = 0.55  # row spacing < 1 packs the bars closer together
    fig, ax = plt.subplots(figsize=(5.0, 0.6 + 0.4 * len(phb) * pitch))
    ax.set_axisbelow(True)  # grid behind the bars
    pn = [p for p, _ in phb][::-1]
    pv = [100 * b / tot_b for _, b in phb][::-1]
    y = np.arange(len(phb)) * pitch
    bars = ax.barh(y, pv, height=0.42, color="#3b738f")
    for bar, s in zip(bars, pv):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                f"{s:.1f}%", va="center", fontsize=8)
    ax.set_yticks(y, pn)
    ax.set_xlabel("share of DRAM bytes moved [%]")
    ax.set_title("WSCMS DRAM traffic by phase")
    plots.save_fig(fig, out / "dram_traffic_by_phase")

    # --- achieved DRAM bandwidth vs the absolute hardware roof: the % of peak in
    # tangible GB/s, with the device's peak as a reference line.
    if peak_gbps > 0:
        pitch = 0.55  # row spacing < 1 packs the bars closer together
        fig, ax = plt.subplots(figsize=(6.0, 0.8 + 0.5 * len(top) * pitch))
        ax.set_axisbelow(True)  # grid behind the bars
        y = np.arange(len(top)) * pitch
        ax.barh(y, [k["gbps"] for k in top][::-1], height=0.42, color="#3b738f")
        ax.axvline(peak_gbps, ls="--", lw=1.2, color="k",
                   label=f"DRAM roof {peak_gbps / 1000:.2f} TB/s")
        ax.set_yticks(y, [k["label"] + libtag(k) for k in top][::-1])
        grey_libs(ax, top[::-1])
        ax.set_xlabel("achieved DRAM bandwidth [GB/s]")
        ax.set_title("Dominant kernels vs the DRAM bandwidth roof")
        ax.legend(fontsize=8, loc="lower right")
        plots.save_fig(fig, out / "achieved_bandwidth")

    # --- launch-config supplement: theoretical occupancy (from the launch shape)
    # and the resource that caps it, for the dominant kernels. When the re-profile
    # added them, also report measured (achieved) occupancy and L2 sector hit rate.
    occ_hdr = ["kernel", "theo_occupancy_pct", "occupancy_limiter",
               "regs_per_thread", "smem_per_block_KB"]
    if have_occ_ach:
        occ_hdr.insert(2, "achieved_occupancy_pct")
    if have_l2:
        occ_hdr.append("l2_hit_pct")
    occ_rows = []
    for k in top:
        row = [k["label"] + libtag(k), f"{k['occupancy_pct']:.0f}"]
        if have_occ_ach:
            row.append(f"{k['occupancy_ach_pct']:.0f}")
        row += [k["limiter"], f"{k['regs']:.0f}", f"{k['smem']:.1f}"]
        if have_l2:
            row.append(f"{k['l2_hit_pct']:.0f}")
        occ_rows.append(row)
    tables.write_table(occ_hdr, occ_rows, out / "occupancy",
                       title="Occupancy and cache behavior")

    weighted_dram = sum(k["dram_pct"] * k["time_ms"] for k in kernels) / \
        sum(k["time_ms"] for k in kernels)

    # Per-ident rows for the cross-bundle aggregate: merge every size variant of a
    # kernel (time-weighted) and keep them all, not just the top 10, so the
    # cross-GPU comparison finds a kernel even when it ranks differently per GPU.
    by_ident: dict[str, dict] = {}
    for k in kernels:
        d = by_ident.setdefault(k["ident"], {"ident": k["ident"], "phase": k["phase"],
                                             "library": k["library"],
                                             "t": 0.0, "dram_w": 0.0, "sm_w": 0.0,
                                             "occ_w": 0.0, "bytes_gb": 0.0,
                                             "flops": 0.0, "full": 0.0,
                                             "has_full": False})
        d["t"] += k["time_ms"]
        d["dram_w"] += k["dram_pct"] * k["time_ms"]
        d["sm_w"] += k["sm_pct"] * k["time_ms"]
        d["occ_w"] += k["occupancy_pct"] * k["time_ms"]
        d["bytes_gb"] += k["bytes_gb"]
        d["flops"] += k["gflops"] * k["time_ms"] * 1e6  # GFLOP/s * ms -> FLOPs
        if k["full_run_pct"] is not None:
            d["full"] += k["full_run_pct"]
            d["has_full"] = True
    ident_rows = [{"ident": d["ident"], "phase": d["phase"], "library": d["library"],
                   "full_run_pct": round(d["full"], 1) if d["has_full"] else None,
                   "dram_pct": round(d["dram_w"] / d["t"], 1),
                   "sm_pct": round(d["sm_w"] / d["t"], 1),
                   "gbps": round(d["bytes_gb"] / (d["t"] / 1e3), 0) if d["t"] else 0,
                   # roofline coordinates (time-weighted across size variants): the
                   # cross-GPU roofline overlay reads these.
                   "gflops": round(d["flops"] / 1e6 / d["t"], 1) if d["t"] else 0.0,
                   "ai": round(d["flops"] / (d["bytes_gb"] * 1e9), 3)
                         if d["bytes_gb"] > 0 else 0.0,
                   "occupancy_pct": round(d["occ_w"] / d["t"], 1)}
                  for d in by_ident.values() if d["t"] > 0]
    ident_rows.sort(key=lambda r: (r["full_run_pct"] if r["full_run_pct"] is not None
                                   else r["dram_pct"]), reverse=True)

    summary = {
        "n_kernels": len(kernels),
        "profiled_time_ms": round(total_s * 1e3, 2),
        "time_weighted_dram_pct": round(weighted_dram, 1),
        "top_kernel": top[0]["label"],
        "top_kernel_dram_pct": round(top[0]["dram_pct"], 1),
        "top_kernel_achieved_GBps": round(top[0]["gbps"], 0),
        "peak_dram_gbps": round(peak_gbps, 0) if peak_gbps else None,
        "peak_fp32_gflops": round(peak_fp32, 0) if peak_fp32 else None,
        # per-ident rows (all kernels, size variants merged) the cross-bundle
        # aggregate uses for the cross-GPU near-roof comparison.
        "kernels": ident_rows,
    }
    if have_flops and peak_fp32 > 0:
        # workload arithmetic intensity: total FLOPs / total DRAM bytes.
        tot_flops = sum(k["gflops"] * k["time_ms"] for k in kernels) * 1e6
        tot_bytes = sum(k["bytes_gb"] for k in kernels) * 1e9
        summary["workload_ai_flop_per_byte"] = (round(tot_flops / tot_bytes, 2)
                                                if tot_bytes > 0 else None)
        summary["roofline_ridge_flop_per_byte"] = round(
            peak_fp32 * 1e9 / (peak_gbps * 1e9), 1) if peak_gbps else None
    if coverage is not None:
        summary["top10_full_run_pct"] = round(coverage, 1)
        summary["top_kernel_full_run_pct"] = round(top[0]["full_run_pct"], 1)
    return summary
