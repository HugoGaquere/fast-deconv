"""Stage 3 -- where the time goes: nsys profile of the first real-data cycle.

Profiles the first cycle of a real deconvolution (example_wscms on a
FastDDFacet dump) under Nsight Systems, exports the per-kernel GPU time
summary, maps every kernel to an algorithm phase by name, and renders the
phase-share breakdown of total GPU time. Real data is used rather than the
synthetic bench because the FFT-vs-clean-loop time split depends on the
data-driven number of minor iterations per scale selection, which the synthetic
fixed-work run cannot reproduce. The top-kernel table feeds the paper directly
and tells the ncu stage which kernels matter.
"""
from __future__ import annotations

import re
import shutil

from harness import (Ctx, SkipStage, first_cycle, iteration_flags, kernel_ident,
                     read_csv_rows, run_cmd, to_num, write_json)

# Ordered first-match rules: lowercase substring of the kernel name -> phase.
# Order matters (DeviceReduce::ArgMax contains both 'argmax' and 'reduce';
# spectral_psf_subtract is a subtraction, not the spectral fit).
PHASE_RULES = [
    ("argmax", "Peak finding (argmax)"),
    ("fftshift", "FFT padding/shift"),
    ("fft", "FFT transforms (cuFFT)"),
    # cuFFT real-transform helpers -- no 'fft' in the name, but pure cuFFT work:
    # preprocess/postprocess_kernel (R2C/C2R pre/post) and kernel_wrapper around
    # packR2C_kernel_Odd_impl / unpackC2R_kernel_Odd_impl.
    ("preprocess_kernel", "FFT transforms (cuFFT)"),
    ("postprocess_kernel", "FFT transforms (cuFFT)"),
    ("kernel_wrapper", "FFT transforms (cuFFT)"),
    ("multiply", "Freq-domain multiply"),
    ("subtract", "PSF subtraction"),
    ("clean_minor", "PSF subtraction"),
    ("spectral", "Spectral fit"),
    ("gemm", "cuBLAS"),
    ("cublas", "cuBLAS"),
    ("mask", "Masking"),
    ("dilation", "Masking"),
    ("threshold", "Masking"),
    ("make_scales", "Scale kernels"),
    ("reduce", "Stats & reductions"),
    ("weighted_", "Stats & reductions"),
    ("mean_residual", "Stats & reductions"),
    ("rms", "Stats & reductions"),
]


def kernel_phase(name: str) -> str:
    low = name.lower()
    for needle, phase in PHASE_RULES:
        if needle in low:
            return phase
    return "Other"


def _short(name: str, n: int = 70) -> str:
    """Demangled kernel names are huge; keep the identifier before the template args."""
    flat = re.sub(r"\s+", " ", name).strip()
    head = flat.split("<")[0].split("(")[0].strip()
    if head.startswith("void "):
        head = head[5:]
    return (head or flat)[:n]


def _stacked_share_bar(plt, items, total, title, base, label_floor=4.0) -> None:
    """One compact 100%-stacked horizontal bar of (name, time) shares -- the
    phase/kernel composition in a single row instead of a tall one-row-per-entry
    chart. Names + exact % live in the legend; only segments wider than
    label_floor get an inline label so the bar stays uncluttered."""
    from analysis import plots
    cmap = plt.get_cmap("tab20")
    # constrained_layout (on globally) collapses this single short bar; the
    # below-axes legend + bbox_inches="tight" handle the layout instead.
    fig, ax = plt.subplots(figsize=(7.0, 1.3))
    fig.set_layout_engine("none")
    left = 0.0
    for i, (name, ms) in enumerate(items):
        s = 100 * ms / total
        ax.barh(0, s, left=left, height=0.6, color=cmap(i % 20),
                edgecolor="white", linewidth=0.5, label=f"{name} ({s:.1f}%)")
        if s >= label_floor:
            ax.text(left + s / 2, 0, f"{s:.0f}%", va="center", ha="center",
                    fontsize=7, color="white")
        left += s
    ax.set_xlim(0, 100)
    ax.set_ylim(-0.3, 0.3)  # match the bar's half-height so it fills the axes
    ax.set_yticks([])
    ax.grid(False)
    ax.set_xlabel("share of total GPU time [%]")
    ax.set_title(title)
    ax.legend(ncol=2, fontsize=6.5, loc="upper center",
              bbox_to_anchor=(0.5, -0.45), frameon=False)
    plots.save_fig(fig, base)


def run(ctx: Ctx) -> dict:
    if shutil.which("nsys") is None:
        raise SkipStage("nsys not found in PATH -- install Nsight Systems to collect "
                        "the phase breakdown")
    if not ctx.dump_dir:
        raise SkipStage("no --dump-dir given -- the phase breakdown profiles the "
                        "first real-data cycle (example_wscms on a FastDDFacet dump)")
    cfg = ctx.preset["nsys"]
    out = ctx.stage_dir("nsys")
    rep = out / "wscms"

    run_cmd(ctx, [
        "nsys", "profile", "--trace=cuda,nvtx", "--force-overwrite=true",
        f"--output={rep}",
        ctx.binary("example_wscms"),
        ctx.dump_dir,
        f"--cycles={first_cycle(ctx.cycles)}",
        f"--device={ctx.device}",
        *iteration_flags(cfg),
    ], "nsys_profile")

    # --force-export regenerates the intermediate SQLite even when one exists
    # from a previous stats call (nsys otherwise refuses and exits with usage).
    run_cmd(ctx, [
        "nsys", "stats", "--report", "cuda_gpu_kern_sum", "--format", "csv",
        f"--output={out / 'stats'}", "--force-overwrite=true", "--force-export=true",
        f"{rep}.nsys-rep",
    ], "nsys_stats")
    # Secondary, version-dependent report; keep it if the tool supports it.
    run_cmd(ctx, [
        "nsys", "stats", "--report", "nvtx_kern_sum", "--format", "csv",
        f"--output={out / 'stats'}", "--force-overwrite=true", "--force-export=true",
        f"{rep}.nsys-rep",
    ], "nsys_stats_nvtx", check=False)

    rows = read_csv_rows(out / "stats_cuda_gpu_kern_sum.csv")
    summary = analyze(rows, out)
    write_json(out / "summary.json", summary)
    return summary


def analyze(rows: list[dict], out) -> dict:
    import matplotlib.pyplot as plt

    from analysis import plots, tables
    plots.setup_style()

    name_col = next(c for c in rows[0] if c.strip() == "Name")
    time_col = next(c for c in rows[0] if c.strip().startswith("Total Time"))
    ns_per_ms = 1e6  # column is "(ns)" in every nsys version that has this report
    kernels = [{"name": r[name_col],
                "short": _short(r[name_col]),
                "ms": float(to_num(r[time_col])) / ns_per_ms,
                "phase": kernel_phase(r[name_col])} for r in rows]
    total_ms = sum(k["ms"] for k in kernels)

    by_phase: dict[str, float] = {}
    for k in kernels:
        by_phase[k["phase"]] = by_phase.get(k["phase"], 0.0) + k["ms"]
    phases = sorted(by_phase.items(), key=lambda kv: kv[1], reverse=True)

    tables.write_table(
        ["phase", "gpu_time_ms", "share_pct"],
        [[p, f"{ms:.2f}", f"{100 * ms / total_ms:.1f}"] for p, ms in phases],
        out / "phase_breakdown", title="GPU time by algorithm phase")

    top = sorted(kernels, key=lambda k: k["ms"], reverse=True)[:15]
    tables.write_table(
        ["kernel", "phase", "gpu_time_ms", "share_pct"],
        [[k["short"], k["phase"], f"{k['ms']:.2f}", f"{100 * k['ms'] / total_ms:.1f}"]
         for k in top],
        out / "top_kernels", title="Top kernels by GPU time")

    _stacked_share_bar(plt, phases, total_ms, "WSCMS minor-cycle GPU time by phase",
                       out / "phase_breakdown", label_floor=4.0)

    # Same data without the phase bucketing: per-kernel shares, with template
    # instantiations / overloads of the same kernel merged. Everything past the
    # top 12 collapses into "Other" so the stacked bar still sums to 100%.
    by_kernel: dict[str, float] = {}
    for k in kernels:
        ident = kernel_ident(k["name"])
        by_kernel[ident] = by_kernel.get(ident, 0.0) + k["ms"]
    ranked = sorted(by_kernel.items(), key=lambda kv: kv[1], reverse=True)
    top_k = ranked[:12]
    other = sum(ms for _, ms in ranked[12:])
    if other > 0:
        top_k = top_k + [("Other", other)]
    _stacked_share_bar(plt, top_k, total_ms, "WSCMS minor-cycle GPU time by kernel",
                       out / "kernel_breakdown", label_floor=3.0)

    return {
        "total_gpu_time_ms": round(total_ms, 1),
        "phase_shares_pct": {p: round(100 * ms / total_ms, 1) for p, ms in phases},
        "top_kernel": top[0]["short"],
        "top_kernel_share_pct": round(100 * top[0]["ms"] / total_ms, 1),
    }
