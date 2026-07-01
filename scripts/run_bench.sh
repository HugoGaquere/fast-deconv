#!/usr/bin/env bash
# Run the WSCMS benchmark sweep and generate marginalized plots.
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "$script_dir/.." && pwd)"

bin="$repo_root/build/Release/bench_wscms"
python="$repo_root/.venv/bin/python"
plot="$script_dir/plot_bench.py"

stamp="$(date +%Y%m%d_%H%M%S)"
outdir="$repo_root/bench_results"
csv="$outdir/bench_$stamp.csv"
plotdir="$outdir/plots_$stamp"
mkdir -p "$outdir"



"$bin" \
  --sizes=1000,2000,4000,6000,8000,10000,12000,14000,16000,18000,20000 \
  --psf-frac=0.085 \
  --nfreq=2,4,6,8,10 \
  --nscales=5,8,10,15 \
  --nfacet=1,50,100 \
  --norder=4 \
  --K=5,10,15,20 \
  --M=250 \
  --runs=10 \
  --warmup=1 \
  --csv="$csv" --force --mode=ofat

"$python" "$plot" "$csv" --outdir="$plotdir" --xscale log

echo "Done. CSV: $csv  Plots: $plotdir"
