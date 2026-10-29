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
#   ./scripts/build.sh                 # host, Release
#   ./scripts/build.sh cuda            # cuda, Release
#   ./scripts/build.sh all             # both backends, Release
#   ./scripts/build.sh cuda Debug      # cuda, Debug
#   ./scripts/build.sh host RelWithDebInfo  # host, Release code + debug info (MAQAO)
#   ./scripts/build.sh --flamegraph    # host, Release, with perf-friendly stacks
#   ./scripts/build.sh --profile       # host, Release, Tracy client linked in
#   ./scripts/build.sh cuda --profile  # cuda, Release, NVTX ranges compiled in
#   ./scripts/build.sh all -t          # both backends, Release, then ctest
#   ./scripts/build.sh -t=Mask         # host, Release, then ctest -R Mask
#   ./scripts/build.sh --python        # host wheel into dist/host
#   ./scripts/build.sh all --python    # both wheels, dist/<backend>/
#   ./scripts/build.sh cuda --clean    # wipe the cuda tree, then rebuild from conan install
#
# conan install and cmake configure run only on a fresh tree; later runs just
# build (CMake still regenerates itself when CMakeLists.txt changes). Pass
# --clean after changing conanfile.py, the Conan profile, -t, --profile or
# --flamegraph.
#
# --python builds the wheel instead of the C++ tree. -t and --flamegraph do not
# apply to it: the test suite is C++/GTest only.
#
# --profile reuses the same tree as a plain build, so toggling it needs --clean. It
# implies the --flamegraph flags: Tracy's sampler needs frame pointers, debug
# info and an exported symbol table (-rdynamic). The client is fetched at
# configure time and must match the server version: v0.14.1, which is what
# `dnf install tracy` provides.
#
# RelWithDebInfo is for MAQAO and other sampling profilers: it builds the same
# -O3 code as Release with -g on top, so hot loops map back to source lines.
# CMake's own RelWithDebInfo default is -O2, hence the override below.
# Dependencies stay on their Release packages; only this project changes.

set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

BACKENDS=(host)
BUILD_TYPE=Release
RUN_TESTS=0
CTEST_FILTER=
BUILD_PYTHON=0
BUILD_FLAMEGRAPH=0
BUILD_PROFILE=0
CLEAN=0

for arg in "$@"; do
  case "$arg" in
    cuda|host)     BACKENDS=("$arg") ;;
    all)           BACKENDS=(cuda host) ;;
    Release|Debug|RelWithDebInfo) BUILD_TYPE="$arg" ;;
    -t|--test)     RUN_TESTS=1 ;;
    -t=*|--test=*) RUN_TESTS=1; CTEST_FILTER="${arg#*=}" ;;
    -p|--python)   BUILD_PYTHON=1 ;;
    -f|--flamegraph) BUILD_FLAMEGRAPH=1 ;;
    --profile)     BUILD_PROFILE=1 ;;
    --clean)       CLEAN=1 ;;
    -h|--help)     sed -n '3,42p' "${BASH_SOURCE[0]}"; exit 0 ;;
    *) echo "usage: $0 [host|cuda|all] [Release|Debug|RelWithDebInfo] [-t[=REGEX]] [--flamegraph] [--profile] [--python] [--clean]" >&2; exit 2 ;;
  esac
done

banner() { printf '\n\033[1;34m========== %s ==========\033[0m\n' "$1"; }

# Backends run in order; with set -e the first failure aborts the rest.
if [ "$BUILD_PYTHON" -eq 1 ] && ! command -v uv >/dev/null; then
  echo "--python needs uv on PATH (the wheel build is driven by scikit-build-core)" >&2
  exit 1
fi

if [ "$BUILD_PYTHON" -eq 1 ] && [ "$BUILD_FLAMEGRAPH" -eq 1 ]; then
  echo "--flamegraph cannot be combined with --python" >&2
  exit 2
fi

if [ "$BUILD_PYTHON" -eq 1 ] && [ "$BUILD_PROFILE" -eq 1 ]; then
  echo "--profile cannot be combined with --python" >&2
  exit 2
fi

CMAKE_CXX_PROFILE_FLAGS=
CMAKE_CUDA_PROFILE_FLAGS=
CMAKE_EXE_PROFILE_LINK_FLAGS=
if [ "$BUILD_FLAMEGRAPH" -eq 1 ] || [ "$BUILD_PROFILE" -eq 1 ]; then
  CMAKE_CXX_PROFILE_FLAGS="-g -fno-omit-frame-pointer"
  CMAKE_CUDA_PROFILE_FLAGS="-g -Xcompiler=-fno-omit-frame-pointer"
  # -rdynamic: without it Tracy's sampler and callstacks resolve to addresses.
  CMAKE_EXE_PROFILE_LINK_FLAGS="-rdynamic"
fi

CMAKE_WITH_PROFILER=OFF
if [ "$BUILD_PROFILE" -eq 1 ]; then
  CMAKE_WITH_PROFILER=ON
fi

# Build dependencies as Release so the cached packages are reused; `&:` scopes
# the RelWithDebInfo setting to this project, which also names the build tree.
CONAN_BUILD_TYPE_ARGS=(-s "build_type=${BUILD_TYPE}")
if [ "$BUILD_TYPE" = RelWithDebInfo ]; then
  CONAN_BUILD_TYPE_ARGS=(-s build_type=Release -s "&:build_type=RelWithDebInfo")
fi

CONAN_WITH_TESTS=False
CONAN_SKIP_TESTS=True
if [ "$RUN_TESTS" -eq 1 ]; then
  CONAN_WITH_TESTS=True
  CONAN_SKIP_TESTS=False
fi

for BACKEND in "${BACKENDS[@]}"; do
  if [ "$BUILD_PYTHON" -eq 1 ]; then
    banner "python wheel (${BACKEND})"
    [ "$CLEAN" -eq 1 ] && rm -rf build/cp*-"${BACKEND}"
    # pyproject.toml picks the backend and its build dir from this; the wheels
    # get per-backend dirs since they share a filename.
    FAST_DECONV_BACKEND="${BACKEND}" uv build --wheel --out-dir "dist/${BACKEND}"
    printf '\n\033[1;32m%s\033[0m\n' "built ${BACKEND} wheel in dist/${BACKEND}"
    continue
  fi

  SUFFIX="$(printf '%s' "$BUILD_TYPE" | tr '[:upper:]' '[:lower:]')-backend_${BACKEND}"
  BUILD_DIR="build/${SUFFIX}"

  [ "$CLEAN" -eq 1 ] && rm -rf "$BUILD_DIR"

  if [ ! -f "$BUILD_DIR/generators/conan_toolchain.cmake" ]; then
    banner "conan install (${BACKEND}, ${BUILD_TYPE})"
    # --build=missing matters for host: the cuda=False emu binary is not prebuilt.
    # Dependencies never build their own tests; `&:` scopes the second -c to this project.
    conan install . -o "fast-deconv/*:backend=${BACKEND}" \
          -o "fast-deconv/*:with_tests=${CONAN_WITH_TESTS}" \
          -c tools.build:skip_test=True \
          -c "&:tools.build:skip_test=${CONAN_SKIP_TESTS}" \
          --build=missing "${CONAN_BUILD_TYPE_ARGS[@]}"
  fi

  if [ ! -f "$BUILD_DIR/CMakeCache.txt" ]; then
    banner "cmake configure (${BACKEND})"
    # FAST_DECONV_BACKEND is passed explicitly: conan puts its cache_variables in
    # the generated preset, not in conan_toolchain.cmake, so configuring against
    # the toolchain alone would silently leave it at its host default.
    cmake -S . -B "$BUILD_DIR" \
          -DCMAKE_TOOLCHAIN_FILE="${REPO}/${BUILD_DIR}/generators/conan_toolchain.cmake" \
          -DFAST_DECONV_BACKEND="$BACKEND" \
          -DBUILD_TESTING="$RUN_TESTS" \
          -DFAST_DECONV_BUILD_TOOLS=ON \
          -DFAST_DECONV_WITH_PROFILER="$CMAKE_WITH_PROFILER" \
          -DCMAKE_BUILD_TYPE="$BUILD_TYPE" \
          -DCMAKE_CXX_FLAGS_RELWITHDEBINFO="-O3 -g -DNDEBUG" \
          -DCMAKE_CUDA_FLAGS_RELWITHDEBINFO="-O3 -g -DNDEBUG" \
          -DCMAKE_CXX_FLAGS="$CMAKE_CXX_PROFILE_FLAGS" \
          -DCMAKE_CUDA_FLAGS="$CMAKE_CUDA_PROFILE_FLAGS" \
          -DCMAKE_EXE_LINKER_FLAGS="$CMAKE_EXE_PROFILE_LINK_FLAGS"
  fi

  banner "cmake build (${BACKEND})"
  cmake --build "$BUILD_DIR" -j

  if [ "$RUN_TESTS" -eq 1 ]; then
    banner "ctest (${BACKEND})"
    ctest --test-dir "$BUILD_DIR" --output-on-failure ${CTEST_FILTER:+-R "$CTEST_FILTER"}
  fi

  printf '\n\033[1;32m%s\033[0m\n' "built ${BACKEND} (${BUILD_TYPE}) in ${BUILD_DIR}"
done
