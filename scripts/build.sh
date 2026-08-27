#!/usr/bin/env bash
#
# Configure and build one or both backends, each into its own tree.
#
# Conan derives the folder from build_type and the backend option, e.g.
# build/release-backend_cuda. We configure against that tree's toolchain rather
# than its CMake preset: a nested tree (a scikit-build wheel build) generates a
# preset of the same name, and CMake refuses to read colliding presets.
#
# Usage:
#   ./scripts/build.sh                 # cuda, Release
#   ./scripts/build.sh host            # host, Release
#   ./scripts/build.sh all             # both backends, Release
#   ./scripts/build.sh cuda Debug      # cuda, Debug
#   ./scripts/build.sh all -t          # both backends, Release, then ctest
#   ./scripts/build.sh --python        # cuda wheel into dist/cuda
#   ./scripts/build.sh all --python    # both wheels, dist/<backend>/
#
# --python builds the wheel instead of the C++ tree. -t does not apply to it:
# the test suite is C++/GTest only.

set -euo pipefail

REPO="$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)"
cd "$REPO"

BACKENDS=(cuda)
BUILD_TYPE=Release
RUN_TESTS=0
BUILD_PYTHON=0

for arg in "$@"; do
  case "$arg" in
    cuda|host)     BACKENDS=("$arg") ;;
    all)           BACKENDS=(cuda host) ;;
    Release|Debug) BUILD_TYPE="$arg" ;;
    -t|--test)     RUN_TESTS=1 ;;
    -p|--python)   BUILD_PYTHON=1 ;;
    -h|--help)     sed -n '3,21p' "${BASH_SOURCE[0]}"; exit 0 ;;
    *) echo "usage: $0 [cuda|host|all] [Release|Debug] [-t] [--python]" >&2; exit 2 ;;
  esac
done

banner() { printf '\n\033[1;34m========== %s ==========\033[0m\n' "$1"; }

# Backends run in order; with set -e the first failure aborts the rest.
if [ "$BUILD_PYTHON" -eq 1 ] && ! command -v uv >/dev/null; then
  echo "--python needs uv on PATH (the wheel build is driven by scikit-build-core)" >&2
  exit 1
fi

for BACKEND in "${BACKENDS[@]}"; do
  if [ "$BUILD_PYTHON" -eq 1 ]; then
    banner "python wheel (${BACKEND})"
    # scikit-build-core reads both of these from the environment. The build dir
    # is per-backend because the two enable different CMake languages and so
    # cannot share a cache; the wheels are too, since they share a filename.
    SKBUILD_CMAKE_DEFINE="FAST_DECONV_BACKEND=${BACKEND}" \
    SKBUILD_BUILD_DIR="build/{wheel_tag}-${BACKEND}" \
      uv build --wheel --out-dir "dist/${BACKEND}"
    printf '\n\033[1;32m%s\033[0m\n' "built ${BACKEND} wheel in dist/${BACKEND}"
    continue
  fi

  SUFFIX="$(printf '%s' "$BUILD_TYPE" | tr '[:upper:]' '[:lower:]')-backend_${BACKEND}"
  BUILD_DIR="build/${SUFFIX}"

  banner "conan install (${BACKEND}, ${BUILD_TYPE})"
  # --build=missing matters for host: the cuda=False emu binary is not prebuilt.
  conan install . -o "fast-deconv/*:backend=${BACKEND}" --build=missing -s "build_type=${BUILD_TYPE}"

  banner "cmake configure (${BACKEND})"
  # FAST_DECONV_BACKEND is passed explicitly: conan puts its cache_variables in
  # the generated preset, not in conan_toolchain.cmake, so configuring against
  # the toolchain alone would silently leave it at its cuda default.
  cmake -S . -B "$BUILD_DIR" \
        -DCMAKE_TOOLCHAIN_FILE="${REPO}/${BUILD_DIR}/generators/conan_toolchain.cmake" \
        -DFAST_DECONV_BACKEND="$BACKEND" \
        -DCMAKE_BUILD_TYPE="$BUILD_TYPE"

  banner "cmake build (${BACKEND})"
  cmake --build "$BUILD_DIR" -j

  if [ "$RUN_TESTS" -eq 1 ]; then
    banner "ctest (${BACKEND})"
    ctest --test-dir "$BUILD_DIR" --output-on-failure
  fi

  printf '\n\033[1;32m%s\033[0m\n' "built ${BACKEND} (${BUILD_TYPE}) in ${BUILD_DIR}"
done
