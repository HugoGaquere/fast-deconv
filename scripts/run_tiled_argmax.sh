#!/usr/bin/env bash
# Run the tiled-argmax tile-size x footprint sweep and generate its plots.
#
# Produces a single CSV (one row per (footprint, tile) pair) that feeds all three
# plotters:
#   plot_tiled_argmax.py            -- 1D tile-size sweep (needs >=2 tiles)
#   plot_tiled_argmax_2d.py         -- tile x footprint heatmaps (needs >=2 of each)
#   plot_tiled_argmax_amortized.py  -- effective speedup vs reseed interval N
#
# Any extra args are forwarded to the benchmark binary, so the defaults below can
# be overridden, e.g.:
#   scripts/run_tiled_argmax.sh --size=8000 --tiles=64,128,256 --reps=20
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "$script_dir/.." && pwd)"

bin="$repo_root/build/Release/bench_tiled_argmax"
python="$repo_root/.venv/bin/python"

stamp="$(date +%Y%m%d_%H%M%S)"
outdir="$repo_root/bench_results"
csv="$outdir/tiled_argmax_$stamp.csv"
plotdir="$outdir/tiled_argmax_plots_$stamp"
mkdir -p "$outdir"

# 2D sweep: every (footprint, tile) pair. amortized_psf picks one footprint for
# the amortization plot (must be one of the swept --psfs values).
amortized_psf=1700

"$bin" \
  --size=20000 \
  --tiles=16,32,64,128,256,512,1024,2048 \
  --psfs=256,512,1024,1700 \
  --reps=50 \
  --warmup=5 \
  --csv="$csv" \
  "$@"

"$python" "$script_dir/plot_tiled_argmax.py" "$csv" --outdir="$plotdir"
"$python" "$script_dir/plot_tiled_argmax_2d.py" "$csv" --outdir="$plotdir"
"$python" "$script_dir/plot_tiled_argmax_amortized.py" "$csv" --psf="$amortized_psf" --outdir="$plotdir"

echo "Done. CSV: $csv  Plots: $plotdir"
