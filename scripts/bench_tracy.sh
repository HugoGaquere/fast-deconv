#!/usr/bin/env bash
#
# Replay the big-ms dump under every thread / binding / memory configuration
# compared in docs/2026-09-29-host-tracy-run-comparison.md, one Tracy capture
# per configuration. The capture starts first and connects as soon as the
# on-demand client comes up; it exits when the replay does.
#
# Usage (from the repo root):
#   ./scripts/bench_tracy.sh                    # every configuration
#   ./scripts/bench_tracy.sh 64_passive spread  # only these configurations
#
# Existing .tracy files are skipped, so an interrupted sweep resumes.
# Overridable: BIN, DATA, CAPTURE, OUT_DIR, PREFIX, CYCLES.

set -uo pipefail

BIN=${BIN:-./build/release-backend_host/replay_ddmsc}
DATA=${DATA:-../../big-ms/dumped_large_2_test-mslist}
CAPTURE=${CAPTURE:-../tracy/build-capture/tracy-capture}
OUT_DIR=${OUT_DIR:-.}
PREFIX=${PREFIX:-run_10}
CYCLES=${CYCLES:-4}

# ducc0 runs inline on the OpenMP threads (nthreads = 1), so its own pool gets no workers.
BASE="DUCC0_NUM_THREADS=1"
PASSIVE="OMP_WAIT_POLICY=passive"
HUGETLB="GLIBC_TUNABLES=glibc.malloc.hugetlb=1"

# name | environment | launcher prefix
CONFIGS=(
  "default|$BASE|"
  "64_active|$BASE OMP_WAIT_POLICY=active OMP_NUM_THREADS=64|"
  "64_active_hugetlb|$BASE OMP_WAIT_POLICY=active OMP_NUM_THREADS=64 $HUGETLB|"
  "64_default|$BASE OMP_NUM_THREADS=64|"
  "48_default_wait|$BASE OMP_NUM_THREADS=48|"
  "96_default_wait|$BASE OMP_NUM_THREADS=96|"
  "32_passive|$BASE $PASSIVE OMP_NUM_THREADS=32|"
  "64_passive|$BASE $PASSIVE OMP_NUM_THREADS=64|"
  "128_passive|$BASE $PASSIVE OMP_NUM_THREADS=128|"
  "64_passive_spread|$BASE $PASSIVE OMP_NUM_THREADS=64 OMP_PROC_BIND=spread OMP_PLACES=cores|"
  "64_passive_close|$BASE $PASSIVE OMP_NUM_THREADS=64 OMP_PROC_BIND=close OMP_PLACES=cores|"
  "64_passive_hugetlb|$BASE $PASSIVE OMP_NUM_THREADS=64 $HUGETLB|"
  "128_passive_hugetlb|$BASE $PASSIVE OMP_NUM_THREADS=128 $HUGETLB|"
  "64_passive_single_socket|$BASE $PASSIVE OMP_NUM_THREADS=64|numactl --cpunodebind=0 --membind=0"
  "128_passive_interleave|$BASE $PASSIVE OMP_NUM_THREADS=128|numactl --interleave=all"
  "64_passive_taskset32|$BASE $PASSIVE OMP_NUM_THREADS=64|taskset -c 0-31,64-95"
  "96_passive_taskset48|$BASE $PASSIVE OMP_NUM_THREADS=96|taskset -c 0-47,64-111"
)

selected() {
  local name=$1; shift
  [[ $# -eq 0 ]] && return 0
  for want in "$@"; do [[ $name == *"$want"* ]] && return 0; done
  return 1
}

for cfg in "${CONFIGS[@]}"; do
  IFS='|' read -r name envs launcher <<<"$cfg"
  selected "$name" "$@" || continue

  out="$OUT_DIR/${PREFIX}_${name}.tracy"
  if [[ -e $out ]]; then
    echo "skip  $name ($out exists)"
    continue
  fi

  echo "run   $name: $envs $launcher"
  "$CAPTURE" -o "$out" -f >"$OUT_DIR/${PREFIX}_${name}.capture.log" 2>&1 &
  capture_pid=$!

  # Word splitting is intended: envs and launcher are space-separated tokens without spaces inside.
  # shellcheck disable=SC2086
  env $envs $launcher "$BIN" "$DATA" --cycle="$CYCLES" >"$OUT_DIR/${PREFIX}_${name}.log" 2>&1
  status=$?

  wait "$capture_pid"
  echo "done  $name (replay exit $status) -> $out"
done
