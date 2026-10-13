#!/usr/bin/env bash
#
# Sequential verification pipeline for fast-deconv, over both backends
# (cuda, host):
#   format -> clang-tidy -> configure/compile -> tests -> device memcheck
#   -> host sanitizers
#
# Run from anywhere; it cd's to the repo root (assumes it lives at repo root or
# is invoked from within the repo). Stages run in order; the first hard failure
# aborts (clang-tidy is advisory and never aborts). The host-sanitizer stage
# always resets FAST_DECONV_SANITIZE=OFF, even on failure.
#
# Usage:
#   ./verify_all.sh                # full pipeline
#   ./verify_all.sh --no-memcheck  # skip the device compute-sanitizer pass
#   ./verify_all.sh --no-sanitize  # skip the slow host ASan/UBSan rebuild+run
#   ./verify_all.sh --no-tidy      # skip clang-tidy

set -euo pipefail

REPO="$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel 2>/dev/null \
        || git rev-parse --show-toplevel)"
cd "$REPO"

RUN_TIDY=1
RUN_MEMCHECK=1
RUN_SANITIZE=1
for arg in "$@"; do
  case "$arg" in
    --no-tidy)     RUN_TIDY=0 ;;
    --no-memcheck) RUN_MEMCHECK=0 ;;
    --no-sanitize) RUN_SANITIZE=0 ;;
    *) echo "unknown flag: $arg" >&2; exit 2 ;;
  esac
done

BACKENDS=(cuda host)

banner() { printf '\n\033[1;34m========== %s ==========\033[0m\n' "$1"; }
note()   { printf '\033[0;36m  %s\033[0m\n' "$1"; }

# Files under review: tracked changes vs HEAD plus untracked, C/C++/CUDA only.
# Filtered to paths that still exist on disk, so deletions/renames in the diff
# (e.g. a removed test) don't get handed to clang-format.
mapfile -t CHANGED < <(
  { git diff --name-only HEAD; git ls-files --others --exclude-standard; } \
    | grep -E '\.(cu|cpp|hpp|h)$' | sort -u \
    | while IFS= read -r f; do [[ -f "$f" ]] && printf '%s\n' "$f"; done || true
)
# clang-tidy needs a compile_commands.json entry, so translation units only.
mapfile -t CHANGED_TU < <(
  printf '%s\n' "${CHANGED[@]:-}" | grep -E '\.(cu|cpp)$' || true
)

# ---------------------------------------------------------------------------
banner "1/7  clang-format (in place, whole file)"
if ((${#CHANGED[@]})); then
  clang-format -i "${CHANGED[@]}"
  reformatted="$(git diff --name-only -- "${CHANGED[@]}" || true)"
  if [[ -n "$reformatted" ]]; then
    echo "$reformatted" | sed 's/^/  reformatted: /'
  else
    note "already formatted"
  fi
else
  note "no changed C/C++/CUDA files"
fi

# ---------------------------------------------------------------------------
banner "2/7  configure Release (conan install + cmake preset, per backend)"
for b in "${BACKENDS[@]}"; do
  conan install . -o "fast-deconv/*:backend=$b" -c tools.build:skip_test=False --build=missing -s build_type=Release
  cmake --preset "conan-release-backend_$b"
done

# ---------------------------------------------------------------------------
banner "3/7  clang-tidy (advisory — does not abort the pipeline)"
if ((RUN_TIDY)) && ((${#CHANGED_TU[@]})); then
  # Host backend sources only exist in the host compile database; everything
  # else (shared + cuda backend) is tidied once against the cuda one.
  for b in "${BACKENDS[@]}"; do
    if [[ $b == host ]]; then
      mapfile -t tus < <(printf '%s\n' "${CHANGED_TU[@]}" | grep '/backend/host/' || true)
    else
      mapfile -t tus < <(printf '%s\n' "${CHANGED_TU[@]}" | grep -v '/backend/host/' || true)
    fi
    ((${#tus[@]})) || continue
    banner "clang-tidy ($b)"
    if clang-tidy -p "build/release-backend_$b" --quiet "${tus[@]}"; then
      note "clang-tidy clean"
    else
      note "clang-tidy reported findings (see above) — pipeline continues"
    fi
  done
elif ((RUN_TIDY)); then
  note "no changed translation units to tidy"
else
  note "skipped (--no-tidy)"
fi

# ---------------------------------------------------------------------------
banner "4/7  compile Release"
for b in "${BACKENDS[@]}"; do
  banner "compile Release ($b)"
  cmake --build "build/release-backend_$b"
done

# ---------------------------------------------------------------------------
banner "5/7  tests (UNIT + NONREG)"
for b in "${BACKENDS[@]}"; do
  banner "tests ($b)"
  ctest --test-dir "build/release-backend_$b" -L UNIT   --no-tests=error --output-on-failure
  ctest --test-dir "build/release-backend_$b" -L NONREG --no-tests=error --output-on-failure
done

# ---------------------------------------------------------------------------
banner "6/7  compute-sanitizer (cuda only: memcheck + racecheck on the UNIT binary)"
if ((RUN_MEMCHECK)); then
  if ! command -v compute-sanitizer >/dev/null; then
    note "compute-sanitizer not on PATH — skipping"
  elif ! command -v nvidia-smi >/dev/null || ! nvidia-smi -L >/dev/null 2>&1; then
    note "no CUDA device — skipping"
  else
    unit_bin="$(find build/release-backend_cuda -type f -name fast_deconv_unit_tests -print -quit)"
    if [[ -z "$unit_bin" ]]; then
      note "unit test binary not found under build/release-backend_cuda — skipping"
    else
      # No --leak-check full: the memory pool retains device allocations by
      # design and would be reported as leaks. memcheck catches OOB/misaligned
      # accesses; racecheck covers shared-memory data races.
      for tool in memcheck racecheck; do
        banner "compute-sanitizer --tool $tool (UNIT binary)"
        compute-sanitizer --tool "$tool" --error-exitcode 1 "$unit_bin"
      done

      # memcheck (only) on the full synthetic run: the NONREG binary is the one
      # thing that exercises the end-to-end minor-cycle orchestration. Run it
      # directly, not through ctest, so its 300s ctest timeout doesn't apply.
      # racecheck is skipped here — too slow on the full run.
      nonreg_bin="$(find build/release-backend_cuda -type f -name fast_deconv_nonreg_tests -print -quit)"
      if [[ -z "$nonreg_bin" ]]; then
        note "nonreg test binary not found under build/release-backend_cuda — skipping"
      else
        banner "compute-sanitizer --tool memcheck (NONREG binary)"
        compute-sanitizer --tool memcheck --error-exitcode 1 "$nonreg_bin"
      fi
    fi
  fi
else
  note "skipped (--no-memcheck)"
fi

# ---------------------------------------------------------------------------
banner "7/7  host sanitizers (ASan + UBSan on UNIT)"
if ((RUN_SANITIZE)); then
  # Always turn the option back off on every tree that was turned on, even if
  # the build or run fails.
  SANITIZED=()
  reset_sanitize() {
    for b in "${SANITIZED[@]}"; do
      banner "reset FAST_DECONV_SANITIZE=OFF ($b)"
      cmake --preset "conan-debug-backend_$b" -DFAST_DECONV_SANITIZE=OFF >/dev/null
    done
  }
  trap reset_sanitize EXIT

  for b in "${BACKENDS[@]}"; do
    banner "host sanitizers ($b)"
    conan install . -o "fast-deconv/*:backend=$b" -c tools.build:skip_test=False --build=missing -s build_type=Debug
    SANITIZED+=("$b")
    cmake --preset "conan-debug-backend_$b" -DFAST_DECONV_SANITIZE=ON
    cmake --build "build/debug-backend_$b"
    ASAN_OPTIONS=protect_shadow_gap=0 \
      LSAN_OPTIONS="suppressions=$REPO/tests/lsan_suppressions.txt" \
      ctest --test-dir "build/debug-backend_$b" -L UNIT --no-tests=error --output-on-failure
  done
else
  note "skipped (--no-sanitize)"
fi

banner "ALL STAGES PASSED"
