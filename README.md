# fast-deconv

GPU-accelerated deconvolution for radio astronomy.

CUDA implementation of the WSCMS minor-cycle loop, designed for integration with [DDFacet](https://github.com/cyriltasse/DDFacet)-style imaging pipelines. The C++/CUDA core is exposed to Python via pybind11.

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

Runtime deps: `numpy`, `cupy-cuda12x`.

## Build (C++ only)

```bash
conan install . --output-folder=build/Release --build=missing -s build_type=Release
cmake --preset conan-release
cmake --build build/Release
ctest --test-dir build/Release
```

Notable outputs: `libfast_deconv.a`, the `example_wscms` driver (runs against FastDDFacet `dump_ref` exports), and the GoogleTest binaries `fast_deconv_unit_tests` / `fast_deconv_nonreg_tests`.

## Tests

```bash
ctest --test-dir build/Release -L UNIT      # unit tests (CPU-oracle based; GPU tests skip if no device)
ctest --test-dir build/Release -L NONREG    # synthetic WSCMS non-regression run vs JSON baseline
```

The non-regression test compares scalar metrics of a full synthetic WSCMS run
against `tests/baselines/wscms_synthetic.json`. After an intentional
algorithmic change, regenerate with `FAST_DECONV_UPDATE_BASELINE=1 ctest
--test-dir build/Release -L NONREG` and commit the reviewed JSON diff.

## Layout

```
include/fast_deconv/   public headers (algorithm, common, core, linalg, matrix, morphology)
src/fast_deconv/       CUDA implementations
src/python/bindings/   pybind11 wrappers
tests/cpp/             GoogleTest unit tests
```

## License

MIT — see [LICENSE](LICENSE).
