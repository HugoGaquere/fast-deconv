# fast-deconv

GPU-accelerated deconvolution for radio astronomy.

CUDA implementation of the DDMSC minor-cycle loop, designed for integration with [DDFacet](https://github.com/cyriltasse/DDFacet)-style imaging pipelines. The C++/CUDA core is exposed to Python via pybind11.

> **Status:** beta (`v0.3.0`). WIP

## Requirements

- CUDA Toolkit 12.x with a compatible NVIDIA GPU
- CMake ≥ 3.26
- [Conan 2.x](https://conan.io) (invoked automatically through `cmake/conan_provider.cmake`)
- Python ≥ 3.10 (for the Python bindings)

The build targets `sm_89` by default; override `CMAKE_CUDA_ARCHITECTURES` for other GPUs.

## Install (Python)

The wheel builds the CUDA extension on install via [scikit-build-core](https://scikit-build-core.readthedocs.io/) and resolves C++ deps with Conan.

```bash
uv sync          # editable install for development
# or
pip install .
```

Runtime deps: `numpy`.

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
conan install . -o "fast-deconv/*:backend=cuda" --build=missing -s build_type=Release
cmake --preset conan-release-backend_cuda
cmake --build build/release-backend_cuda
ctest --test-dir build/release-backend_cuda
```

**The `host` backend does not compute anything yet.** Every kernel is a stub
that throws `"host backend not implemented yet"`; only the CPU-only tests
(`test_convergence`) run, and the CUDA apps, benches and tests are skipped. It
exists so the backend seam stays honest — build it to check that a change keeps
the tree backend-agnostic, not to run a deconvolution.

Notable outputs: `libfast_deconv.a`, the `example_ddmsc` driver (runs against FastDDFacet `dump_ref` exports), and the GoogleTest binaries `fast_deconv_unit_tests` / `fast_deconv_nonreg_tests`.

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
