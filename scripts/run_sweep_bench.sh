#!/usr/bin/env bash
#
# Image-size scaling sweep of `replay_ddmsc --bench` over one dumped cycle.
#
# One replay_ddmsc process per size (RUNS runs each), each writing its own CSV
# and log; the CSVs are merged into bench.csv at the end.
#
# Usage:
#   ./scripts/run_sweep_bench.sh <dump_dir> <cycle> [extra replay_ddmsc flags...]
#
# Environment (defaults in brackets):
#   BIN     replay_ddmsc binary          [<repo>/build/release-backend_cuda/replay_ddmsc]
#   DEVICE  CUDA device index            [0]
#   SIZES   crop sizes, "full" = no crop [2000 4000 8000 12000 16000 full]
#   OUTER   outer iterations per run     [200]
#   RUNS    runs per size (--runs)       [5]
#   LABEL   CSV label                    [GPU name from nvidia-smi]
#   OUTDIR  output directory             [bench_<label>_cycle<cycle>_<timestamp>]

set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 <dump_dir> <cycle> [extra replay_ddmsc flags...]" >&2
  exit 2
fi
DUMP="$1"
CYCLE="$2"
shift 2

BIN="${BIN:-$REPO/build/release-backend_cuda/replay_ddmsc}"
DEVICE="${DEVICE:-0}"
SIZES="${SIZES:-2000 4000 8000 12000 16000 full}"
OUTER="${OUTER:-200}"
RUNS="${RUNS:-5}"
# CUDA numbers GPUs fastest-first, nvidia-smi by PCI bus: make them agree.
export CUDA_DEVICE_ORDER=PCI_BUS_ID
# With CUDA_VISIBLE_DEVICES set, --device indexes that list, not the physical GPUs.
GPU="$DEVICE"
[[ -n "${CUDA_VISIBLE_DEVICES:-}" ]] && GPU="$(cut -d, -f$((DEVICE + 1)) <<<"$CUDA_VISIBLE_DEVICES")"
echo "--device=$DEVICE is nvidia-smi GPU $GPU: $(nvidia-smi --query-gpu=name,pci.bus_id --format=csv,noheader -i "$GPU")"
LABEL="${LABEL:-$(nvidia-smi --query-gpu=name --format=csv,noheader -i "$GPU" | tr ' ' '_')}"
OUTDIR="${OUTDIR:-bench_${LABEL}_cycle${CYCLE}_$(date +%y%m%d_%H%M%S)}"
mkdir -p "$OUTDIR"

for size in $SIZES; do
  crop=()
  [[ "$size" == full ]] || crop=("--crop=$size")
  name="$OUTDIR/crop${size}"
  echo "=== $size ($(date +%T))"
  "$BIN" "$DUMP" --device="$DEVICE" --cycle="$CYCLE" --bench --outer="$OUTER" --runs="$RUNS" \
    "${crop[@]}" --csv="$name.csv" --label="$LABEL" "$@" > "$name.log" 2>&1
  grep -E '^run [0-9]' "$name.log"
done

# Merge: one header, then every size's rows.
first=1
for f in "$OUTDIR"/crop*.csv; do
  if [[ $first -eq 1 ]]; then cat "$f"; first=0; else tail -n +2 "$f"; fi
done > "$OUTDIR/bench.csv"
echo "merged into $OUTDIR/bench.csv"
