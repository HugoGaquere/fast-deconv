# Paper-data harness

One-command collection of every measurement for the SPIE paper results section
(GPU DDMSC deconvolution). Designed to be carried to any GPU machine: build the
repo, run one command, get back a portable, self-describing bundle directory.
Bundles from several machines are then merged into the paper figures/tables
with a second command.

No CPU-baseline comparison is collected by design: performance claims are
anchored in hardware-relative terms (bandwidth utilization, scaling exponents,
absolute ms per minor iteration).

## Prerequisites

- Built binaries (`bench_ddmsc`, `bench_tiled_argmax`, `example_ddmsc`):

  ```bash
  conan install . --output-folder=build/Release --build=missing -s build_type=Release
  cmake --preset conan-release
  cmake --build build/Release -j
  ```

  The build uses `CMAKE_CUDA_ARCHITECTURES=native`: it targets whatever GPUs
  are visible at *configure* time, so configure each machine's build tree on
  that machine (don't copy a configured `build/` across hosts). Only a GPU-less
  build node needs an explicit `-DCMAKE_CUDA_ARCHITECTURES=NN`.

- Python ≥ 3.10 with `numpy` and `matplotlib` (the repo venv has them:
  `.venv/bin/python`).
- `nvidia-smi` in PATH (machine manifest; clock locking).
- Optional: `nsys` (phase breakdown stage), `ncu` (roofline stage). Missing
  tools skip their stage cleanly. `ncu` needs performance-counter permission:
  run as root or set the kernel module option
  `NVreg_RestrictProfilingToAdminUsers=0`.

## Usage

```bash
# Sanity pass (~10 min): checks binaries, tools, parsing, plotting.
.venv/bin/python scripts/paper/paper.py run --preset quick

# Paper numbers (hours, dominated by the scaling sweep).
.venv/bin/python scripts/paper/paper.py run --preset full

# With real DDFacet dump_ref data: required for the phase breakdown (nsys),
# roofline (ncu), cycle timings + convergence (realdata) -- all profile real data.
.venv/bin/python scripts/paper/paper.py run --preset full \
    --dump-dir /data/dump_ref --cycles 1-6

# Re-run a subset into an existing bundle (stages are independent):
.venv/bin/python scripts/paper/paper.py run --stages ncu,nsys \
    --dump-dir /data/dump_ref --bundle paper_data/a100_20260611_120000

# Merge bundles from several machines into paper-ready output:
.venv/bin/python scripts/paper/paper.py aggregate paper_data/* --outdir paper_figures

# Rebuild tables/figures from raw data already in bundles (after analysis-code
# changes; no GPU or rebuilt binaries needed):
.venv/bin/python scripts/paper/paper.py reanalyze paper_data/*
```

On shared servers, selecting the GPU via the environment works as expected —
all child processes (benches, nsys, ncu) inherit it, and the machine manifest /
clock locking map the index back to the right physical GPU for nvidia-smi:

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=4 \
    python scripts/paper/paper.py run --preset full
```

(`--device` then stays at its default 0, meaning "first visible device", just
like the CUDA runtime. Integer `CUDA_VISIBLE_DEVICES` entries should be used
with `CUDA_DEVICE_ORDER=PCI_BUS_ID` so they match nvidia-smi's numbering.)

Useful flags: `--device N`, `--bin-dir PATH`, `--label TAG` (e.g.
`--label before` / `--label after` for the tiled-argmax integration A/B — the
aggregate step overlays bundles regardless of whether they differ by GPU or by
label), `--lock-clocks` (pins GPU core clocks for low-noise A/B runs; needs
root), `--dump-gpu-output` (write per-cycle components + residual; the residual
is `n_freq × nrow × ncol` float32, i.e. multi-GiB on survey-size images).

## Stages and what they produce

| stage      | tool               | output (per bundle)                                       | paper use |
|------------|--------------------|-----------------------------------------------------------|-----------|
| `scaling`  | `bench_ddmsc`      | `scaling/bench.csv`, scaling panels, fitted exponents     | wall time / memory ∝ N^p, throughput, parameter sensitivity |
| `argmax`   | `bench_tiled_argmax` | tile sweep, speedup heatmap, amortized-speedup curve    | optimization study |
| `nsys`     | Nsight Systems on `example_ddmsc` | per-kernel GPU time, phase-share breakdown of the first real-data cycle | where the time goes (Amdahl); needs `--dump-dir` |
| `ncu`      | Nsight Compute on `example_ddmsc` | per-kernel achieved GB/s and % of peak DRAM/SM (opening launches of the first real-data cycle), cross-referenced with each kernel's nsys time share | efficiency without a CPU baseline; shows the profiled kernels dominate the runtime; needs `--dump-dir` |
| `realdata` | `example_ddmsc`    | per-cycle timings, ms/iter, convergence plot              | real-workload numbers (needs `--dump-dir`) |
| `fidelity` | numpy              | component + residual comparison vs DDFacet reference      | correctness (needs `--ref-dir`, see below) |

Every stage writes raw data (CSV / .nsys-rep / .ncu-rep), figures as PNG + PDF,
and a `summary.json` with its headline numbers. `env.json` records GPU, driver,
CUDA, tool versions, and the git commit. All subprocess output is kept under
`logs/`.

## Bundle layout

```
paper_data/<gpu>_<label>_<stamp>/
  env.json            machine manifest
  summary.json        per-stage status + headline numbers
  logs/               full output of every command
  scaling/  argmax/  nsys/  ncu/  realdata/  fidelity/
```

## Reference outputs for the fidelity stage (not available yet)

When the DDFacet-side exports exist, point `--ref-dir` at:

```
ref_dir/
  cycle_<N>/
    components.csv   # header: row,col,scale,gain[,coeff0..coeffK]
    residual.npy     # float32 (n_freq, nrow, ncol), residual after the minor cycle
```

This is exactly the format `example_ddmsc --dump-result=DIR` writes for the GPU
side, so the FastDDFacet exporter just needs to target it. Until then the
fidelity stage reports `skipped`.
