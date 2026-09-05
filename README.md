# fast-deconv

GPU-accelerated deconvolution for radio astronomy.

CUDA implementation of the DDMSC minor-cycle loop, designed for integration with [DDFacet](https://github.com/cyriltasse/DDFacet)-style imaging pipelines. The C++/CUDA core is exposed to Python via pybind11.

> **Status:** beta (`v0.7.0`). WIP

## Requirements

- CUDA Toolkit with a compatible NVIDIA GPU (CUDA backend only)
- OpenMP-capable C++ compiler (host backend)
- CMake ≥ 3.26
- [Conan 2.x](https://conan.io) (invoked automatically through `cmake/conan_provider.cmake`)
- Python ≥ 3.10 (for the Python bindings)

CUDA builds target the local GPU (`native`) by default. Set `CMAKE_CUDA_ARCHITECTURES`
(or `CUDAARCHS`) explicitly for other GPUs, distribution builds, or machines without a GPU.

## Install (Python)

The wheel builds the CUDA extension on install via [scikit-build-core](https://scikit-build-core.readthedocs.io/) and resolves C++ deps with Conan.

```bash
uv sync          # editable install for development
# or
pip install .
```

Runtime deps: `numpy`. Linux wheels bundle linked shared dependencies, including
OpenMP and CUDA math libraries where used. The platform C/C++ runtime and NVIDIA
driver remain external. Publishing portable wheels still requires a suitable
build environment and platform wheel repair (handled by cibuildwheel).

## Build (C++ only)

`FAST_DECONV_BACKEND` selects the compute backend: `cuda` (default) or `host`.
Each backend gets its own tree, so the two never clobber each other.

```bash
./scripts/build.sh              # cuda, Release
./scripts/build.sh host         # host, Release
./scripts/build.sh all          # both backends, Release
./scripts/build.sh all -t       # both backends, Release, then ctest
./scripts/build.sh --python     # cuda wheel into dist/cuda
./scripts/build.sh all --python # both wheels, into dist/<backend>/
```

Or by hand:

```bash
conan install . -o "fast-deconv/*:backend=cuda" -c tools.build:skip_test=False --build=missing -s build_type=Release
cmake --preset conan-release-backend_cuda
cmake --build build/release-backend_cuda
ctest --test-dir build/release-backend_cuda
```

The `host` backend runs the same public API on the CPU, using PocketFFT and OpenMP.

Conan is a dependency provider; this recipe does not package the C++ library.
For direct CMake builds, `FAST_DECONV_BUILD_TOOLS` and
`FAST_DECONV_BUILD_BENCHMARKS` default to `OFF`. Enable them to build
`replay_ddmsc` and the CUDA benchmarks, respectively. `scripts/build.sh` enables
both and only builds/resolves tests when `-t` is supplied.

CMake uses `BUILD_TESTING` (replacing `BUILD_TESTS`). For manual Conan installs,
`-o "fast-deconv/*:with_tests=False"` or `-c tools.build:skip_test=True` omits test
dependencies and disables testing in the generated preset. When configuring with
the toolchain file directly, also pass the matching `-DBUILD_TESTING=OFF`; Conan's
cache variables are applied by its presets. Python builds always omit C++ tests.

Notable outputs: `libfast_deconv.a`, the optional `replay_ddmsc` driver, and the
GoogleTest binaries `fast_deconv_unit_tests` / `fast_deconv_nonreg_tests` when enabled.

## Tests

```bash
ctest --test-dir build/release-backend_cuda -L UNIT      # unit tests (CPU-oracle based; GPU tests skip if no device)
ctest --test-dir build/release-backend_cuda -L NONREG    # synthetic DDMSC non-regression run vs JSON baseline
```

The non-regression test compares scalar metrics of a full synthetic DDMSC run
against `tests/baselines/ddmsc_synthetic.json`. After an intentional
algorithmic change, regenerate with `FAST_DECONV_UPDATE_BASELINE=1 ctest
--test-dir build/release-backend_cuda -L NONREG` and commit the reviewed JSON diff.

## Layout

```
include/fast_deconv/   public headers (algorithm, common, core, linalg, matrix, morphology)
src/fast_deconv/       shared sources; backend/<cuda|host>/ per-backend implementations
src/python/bindings/   pybind11 wrappers
tests/cpp/             GoogleTest unit tests
```

## License

MIT — see [LICENSE](LICENSE).
