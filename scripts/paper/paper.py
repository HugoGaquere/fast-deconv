#!/usr/bin/env python3
"""One-command data collection for the SPIE paper (GPU WSCMS deconvolution).

Run everything this machine can produce into a self-describing bundle:

    .venv/bin/python scripts/paper/paper.py run --preset full

Then copy the bundles from every GPU machine somewhere central and build the
cross-GPU figures and tables:

    .venv/bin/python scripts/paper/paper.py aggregate paper_data/* --outdir paper_figures

See README.md in this directory for prerequisites, stage details, runtimes,
and the expected layout of the (future) DDFacet reference outputs.
"""
from __future__ import annotations

import argparse
import datetime
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from harness import Ctx, SkipStage, try_capture, write_json  # noqa: E402
from stages import env_info  # noqa: E402

# --------------------------------------------------------------------------- #
#  Presets
# --------------------------------------------------------------------------- #

PRESETS = {
    # Paper numbers. Scaling sweep is the long pole (hours on a big GPU);
    # oversize configs are OOM-skipped automatically on smaller cards.
    "full": {
        "scaling": dict(
            mode="ofat",
            sizes=[1000, 2000, 4000, 6000, 8000, 10000, 12000, 14000, 16000,
                   18000, 20000],
            nfreq=[2, 4, 6, 8, 10], nscales=[5, 8, 10, 15], nfacet=[1, 50, 100],
            norder=[4], psf_frac=[0.085], K=[5, 10, 15, 20], M=[250],
            runs=10, warmup=2),
        "argmax": dict(size=20000, tiles=[16, 32, 64, 128, 256, 512, 1024, 2048],
                       psfs=[256, 512, 1024, 1700], reps=50, warmup=5),
        # nsys/ncu profile the first real-data cycle (example_wscms on --dump-dir),
        # not the synthetic bench. nsys runs the full cycle so its time-share is
        # representative (capping inner iters would re-bias the FFT/clean ratio);
        # ncu bounds the run to 1 scale selection x 5 clean iters -- enough to
        # hit every kernel a few times, since efficiency is iteration-invariant.
        # No launch-count cap: the iteration bound already makes the run short, and
        # a cap truncates mid-scale-selection (>500 cuFFT launches) before the
        # clean-loop kernels appear.
        "nsys": dict(),
        "ncu": dict(max_clean_iter=5, max_iter=5),
    },
    # ~10 minute sanity pass: verifies binaries, tools, parsing, and plotting.
    "quick": {
        "scaling": dict(
            mode="ofat",
            sizes=[1000, 2000, 4000], nfreq=[2, 4, 8], nscales=[5], nfacet=[1],
            norder=[2], psf_frac=[0.085], K=[2], M=[50], runs=3, warmup=1),
        "argmax": dict(size=8000, tiles=[64, 128, 256, 512], psfs=[512, 1024],
                       reps=10, warmup=2),
        "nsys": dict(),
        "ncu": dict(max_clean_iter=5, max_iter=5),
    },
}

STAGE_ORDER = ["scaling", "argmax", "nsys", "ncu", "realdata", "fidelity"]


def _stage_fn(name):
    # Imported lazily so a missing optional dependency in one stage doesn't
    # block the others.
    if name == "scaling":
        from stages import scaling
        return scaling.run
    if name == "argmax":
        from stages import argmax
        return argmax.run
    if name == "nsys":
        from stages import nsys_breakdown
        return nsys_breakdown.run
    if name == "ncu":
        from stages import ncu_roofline
        return ncu_roofline.run
    if name == "realdata":
        from stages import realdata
        return realdata.run
    if name == "fidelity":
        from stages import fidelity
        return fidelity.run
    raise KeyError(name)


# --------------------------------------------------------------------------- #
#  Clock locking (best effort; needs root)
# --------------------------------------------------------------------------- #

def lock_clocks(device: int) -> bool:
    smi_id = env_info.smi_index(device)
    out = try_capture(["nvidia-smi", "--query-gpu=clocks.max.sm",
                       "--format=csv,noheader,nounits", "-i", smi_id])
    if out is None:
        return False
    mhz = out.strip().splitlines()[0].strip()
    ok = try_capture(["nvidia-smi", "-i", smi_id, "-lgc", f"{mhz},{mhz}"])
    if ok is None:
        print("  [warn] could not lock GPU clocks (needs root); continuing unlocked")
        return False
    print(f"  locked GPU {smi_id} core clock at {mhz} MHz")
    return True


def unlock_clocks(device: int) -> None:
    try_capture(["nvidia-smi", "-i", env_info.smi_index(device), "-rgc"])


# --------------------------------------------------------------------------- #
#  Commands
# --------------------------------------------------------------------------- #

def cmd_run(args) -> int:
    repo_root = Path(__file__).resolve().parents[2]
    bin_dir = Path(args.bin_dir) if args.bin_dir else repo_root / "build" / "Release"

    print("Collecting machine info...")
    env = env_info.collect(repo_root, args.device)
    env["label"] = args.label
    env["preset"] = args.preset
    gpu_name = env.get("gpu", {}).get("name", "")
    print(f"  GPU {args.device}: {gpu_name or 'unknown (nvidia-smi not found?)'}")

    if args.bundle:
        bundle = Path(args.bundle)
    else:
        stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        parts = [env_info.gpu_slug(env)] + ([args.label] if args.label else []) + [stamp]
        bundle = Path(args.outdir) / "_".join(parts)
    bundle.mkdir(parents=True, exist_ok=True)
    write_json(bundle / "env.json", env)
    print(f"  bundle: {bundle}")

    ctx = Ctx(repo_root=repo_root, bin_dir=bin_dir, bundle=bundle,
              device=args.device, preset_name=args.preset,
              preset=PRESETS[args.preset], dump_dir=args.dump_dir or "",
              ref_dir=args.ref_dir or "", cycles=args.cycles,
              label=args.label, dump_gpu_output=args.dump_gpu_output,
              force_auto_mask_last=args.force_auto_mask_last, env=env)

    stage_names = ([s.strip() for s in args.stages.split(",")] if args.stages
                   else STAGE_ORDER)
    unknown = [s for s in stage_names if s not in STAGE_ORDER]
    if unknown:
        print(f"Unknown stage(s): {', '.join(unknown)} (known: {', '.join(STAGE_ORDER)})")
        return 1

    locked = lock_clocks(args.device) if args.lock_clocks else False
    results: dict = {}
    try:
        for name in stage_names:
            print(f"\n===== stage: {name} =====")
            t0 = time.time()
            try:
                summary = _stage_fn(name)(ctx) or {}
                status = "ok"
            except SkipStage as e:
                summary, status = {"note": str(e)}, "skipped"
                print(f"  skipped: {e}")
            except Exception as e:  # keep going: unattended multi-hour runs
                summary, status = {"error": str(e)}, "failed"
                print(f"  FAILED: {e}")
            results[name] = {"status": status,
                             "seconds": round(time.time() - t0, 1), **summary}
            write_json(bundle / "summary.json", results)
    finally:
        if locked:
            unlock_clocks(args.device)

    print(f"\n===== done =====\nBundle: {bundle}")
    for name, r in results.items():
        print(f"  {name:10s} {r['status']:8s} ({r['seconds']}s)")
    print("\nNext: copy the bundle to where the other GPUs' bundles live and run\n"
          f"  python scripts/paper/paper.py aggregate <bundles...> --outdir paper_figures")
    return 1 if any(r["status"] == "failed" for r in results.values()) else 0


def cmd_aggregate(args) -> int:
    from analysis import aggregate
    aggregate.run(args.bundles, args.outdir)
    return 0


def cmd_reanalyze(args) -> int:
    """Rebuild tables/figures from raw data already in bundles (no GPU needed)."""
    from harness import read_csv_rows, read_json, to_num

    def refresh(results, name, summary):
        old = results.get(name, {})
        results[name] = {"status": old.get("status", "ok"),
                         "seconds": old.get("seconds", 0.0), **summary}

    for b in args.bundles:
        bundle = Path(b)
        summary_path = bundle / "summary.json"
        results = read_json(summary_path) if summary_path.exists() else {}
        done = []

        scaling_csv = bundle / "scaling" / "bench.csv"
        if scaling_csv.exists():
            from stages import scaling
            rows = [{k: to_num(v) for k, v in r.items()}
                    for r in read_csv_rows(scaling_csv)]
            s = scaling.analyze(rows, bundle / "scaling")
            write_json(bundle / "scaling" / "summary.json", s)
            refresh(results, "scaling", s)
            done.append("scaling")

        argmax_csv = bundle / "argmax" / "tiled_argmax.csv"
        if argmax_csv.exists():
            from stages import argmax
            rows = [{k: to_num(v) for k, v in r.items()}
                    for r in read_csv_rows(argmax_csv)]
            s = argmax.analyze(rows, bundle / "argmax")
            write_json(bundle / "argmax" / "summary.json", s)
            refresh(results, "argmax", s)
            done.append("argmax")

        nsys_csv = bundle / "nsys" / "stats_cuda_gpu_kern_sum.csv"
        if nsys_csv.exists():
            from stages import nsys_breakdown
            s = nsys_breakdown.analyze(read_csv_rows(nsys_csv), bundle / "nsys")
            write_json(bundle / "nsys" / "summary.json", s)
            refresh(results, "nsys", s)
            done.append("nsys")

        raw_csv = bundle / "ncu" / "raw.csv"
        if raw_csv.exists():
            from stages import ncu_roofline
            s = ncu_roofline.analyze(raw_csv.read_text(), bundle / "ncu", nsys_csv)
            write_json(bundle / "ncu" / "summary.json", s)
            refresh(results, "ncu", s)
            done.append("ncu")

        cycles_csv = bundle / "realdata" / "cycles.csv"
        if cycles_csv.exists():
            from stages import realdata
            rows = [{k: to_num(v) for k, v in r.items()}
                    for r in read_csv_rows(cycles_csv)]
            s = realdata.analyze(rows, bundle / "realdata")
            write_json(bundle / "realdata" / "summary.json", s)
            refresh(results, "realdata", s)
            done.append("realdata")

        if done:
            write_json(summary_path, results)
        print(f"{bundle}: reanalyzed {', '.join(done) if done else 'nothing (no raw data)'}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)

    rp = sub.add_parser("run", help="collect all paper data on this machine")
    rp.add_argument("--preset", choices=sorted(PRESETS), default="full")
    rp.add_argument("--stages", default="",
                    help=f"comma-separated subset of: {','.join(STAGE_ORDER)}")
    rp.add_argument("--bin-dir", default="",
                    help="directory with the built binaries (default: build/Release)")
    rp.add_argument("--device", type=int, default=0)
    rp.add_argument("--outdir", default="paper_data",
                    help="where new bundles are created")
    rp.add_argument("--bundle", default="",
                    help="re-run stages into an existing bundle directory")
    rp.add_argument("--label", default="",
                    help="tag for this run (e.g. 'before'/'after' for an A/B)")
    rp.add_argument("--dump-dir", default="",
                    help="FastDDFacet dump_ref export (enables the realdata stage)")
    rp.add_argument("--cycles", default="1",
                    help="cycle spec for example_wscms, e.g. '1,2,4-6'")
    rp.add_argument("--ref-dir", default="",
                    help="DDFacet reference outputs (enables the fidelity stage)")
    rp.add_argument("--dump-gpu-output", action="store_true",
                    help="write GPU components+residual per cycle (large files)")
    rp.add_argument("--force-auto-mask-last", action="store_true",
                    help="force auto-masking on the last cycle of the set")
    rp.add_argument("--lock-clocks", action="store_true",
                    help="lock GPU core clocks during the run (needs root)")
    rp.set_defaults(fn=cmd_run)

    agp = sub.add_parser("aggregate", help="merge bundles into paper figures/tables")
    agp.add_argument("bundles", nargs="+", help="bundle directories from `run`")
    agp.add_argument("--outdir", default="paper_figures")
    agp.set_defaults(fn=cmd_aggregate)

    rean = sub.add_parser("reanalyze", help="rebuild tables/figures from raw data "
                          "already in bundles (after analysis changes; no GPU)")
    rean.add_argument("bundles", nargs="+", help="bundle directories from `run`")
    rean.set_defaults(fn=cmd_reanalyze)

    args = ap.parse_args()
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
