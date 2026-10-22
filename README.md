# fast-deconv

GPU-accelerated DDMSC deconvolution for radio astronomy, callable from Python.

fast-deconv is a high-performance multi-scale and multi-frequency deconvolution engine with several execution backends: CUDA for NVIDIA GPUs and OpenMP for CPUs.
It's designed for integration with [DDFacet](https://github.com/cyriltasse/DDFacet)-style imaging pipelines.

> **Status:** beta (`v0.8.0`). The API may still change between minor versions.

## Quick start

```bash
pip install .
```

```python
import numpy as np
from fast_deconv import DDMSC

# Demo only: every input below is synthetic.
n_facets, n_freq, n = 1, 4, 256

# One delta PSF per (facet, frequency)
psfs = np.zeros((n_facets, n_freq, n, n), np.float32)
psfs[:, :, n // 2, n // 2] = 1.0

# Spectral model [1, ν/ν0] per channel
nu = np.array([0.85, 0.95, 1.05, 1.15])                      # ν / ν0 of each channel
xdes = np.vander(nu, 2, increasing=True).astype(np.float32)  # (n_freq, n_order)

deconv = DDMSC(
    raw_psfs=psfs,
    xdes=xdes,
    scale_mask=np.zeros((n, n), bool),                      # True = masked out
    scale_sigmas=np.array([0.0, 1.5], np.float32),          # sigma 0 = point source
    scale_bias=np.array([1.0, 0.8], np.float32),
    map_pixel_facet=np.zeros((n, n), np.int32),             # facet index per pixel
    dirty_nrow=n,
    dirty_ncol=n,
)
deconv.gamma = 0.1
deconv.flux_threshold = 0.05   # stop once the residual peak drops below this

dirty = np.zeros((n_freq, n, n), np.float32)   # fake sky: one point source
dirty[:, 100, 120] = 5.0
weights = np.full(n_freq, 1 / n_freq, np.float32)   # relative band weights, sum to 1
result = deconv.run(dirty, np.ones_like(dirty), weights)

# ConvergenceStatus.converged 57 0.0123...
# [(100, 120), (100, 120), (100, 120)] (57, 2)
print(result.status, result.n_components, result.final_flux)
print(result.peak_coords[:3], result.coeffs.shape)

# dirty now holds the residual: the 5.0 peak is down to 0.0123...
print(dirty.max())
```

## Python API

`DDMSC` is a stateful session. Create it once per imaging run, then call `run()`
once per major cycle.

| Constructor argument | Shape | dtype | Meaning |
|---|---|---|---|
| `raw_psfs` | `(n_facets, n_freq, psf_h, psf_w)` | `float32` | Stokes-I PSF of each facet and band, normalised to unit peak |
| `xdes` | `(n_freq, n_order)` | `float32` | Spectral basis evaluated at each band, e.g. `(ν/ν0)^k`; any linear basis, up to 8 terms |
| `scale_mask` | `(nrow, ncol)` | `bool` | External mask, `True` = pixel excluded from cleaning |
| `scale_sigmas` | `(n_scales,)` | `float32` | Gaussian sigma of each scale, in pixels; `0` = delta scale |
| `scale_bias` | `(n_scales,)` | `float32` | Per-scale bias multiplying the peak during scale selection ([Offringa & Smirnov 2017](https://arxiv.org/abs/1706.06786)); `1` for the delta scale |
| `map_pixel_facet` | `(nrow, ncol)` | `int32` | Facet (= `raw_psfs` axis 0) index of every pixel; all zeros for a single PSF |
| `dirty_nrow`, `dirty_ncol` | | `int` | Dirty image size |
| `exec_device` | | `int` | CUDA device index, default `0` |

| `run()` argument | Shape | dtype | Meaning |
|---|---|---|---|
| `dirty` | `(n_freq, nrow, ncol)` | `float32` | Stokes-I residual cube, overwritten with the new residual |
| `jones_norm` | `(n_freq, nrow, ncol)` | `float32` | Jones-amplitude normalisation per band and pixel, used in the spectral fit; ones without beam or DD correction |
| `weights_freq` | `(n_freq,)` | `float32` | Relative band weights, summing to 1 |

`run()` returns a `DDMSCResult`:

- **Components:** `peak_coords`, `scales`, `gains`, `coeffs`, `n_components`
- **Flux and noise:** `initial_flux`, `initial_rms`, `final_flux`, `final_rms`, `stop_flux`
- **Iterations:** `total_iterations`, `outer_iterations`
- **Stop reason:** `status` (`ConvergenceStatus`)

Algorithm parameters are properties you can change between `run()` calls:

- **Clean loop:** `gamma`, `peak_factor`, `max_sub_iteration`, `clean_negative`
- **Outer loop:** `max_iteration`, `divergence_factor`, `stall_threshold`
- **Stopping:** `flux_threshold`, `stop_rms_factor`, `stop_peak_factor`, `stop_cycle_factor`, `stop_sidelobe_level`
- **Auto-mask:** `auto_mask`, `force_auto_mask`, `auto_mask_peak_threshold`, `auto_mask_rms_threshold`

Types are in `src/fast_deconv/_fast_deconv/ddmsc.pyi`.

## Requirements

- Linux
- CMake ≥ 3.26
- A C++20 compiler with OpenMP
- [Conan 2.x](https://conan.io). The build calls it automatically through
  `cmake/conan_provider.cmake`.
- Python ≥ 3.10, for the Python package
- **CUDA backend only:** the CUDA Toolkit and an NVIDIA GPU

CUDA builds target the GPU in the build machine (`native`) by default. To build for
other GPUs, or on a machine without one, set `CMAKE_CUDA_ARCHITECTURES` or
`CUDAARCHS`, e.g. `CUDAARCHS="80;90"`.

## Installing the Python package

scikit-build-core compiles the extension during install, and Conan fetches the C++
dependencies. NumPy is the only runtime dependency.

```bash
pip install .     # CUDA backend
uv sync           # editable development install; rebuilds on import after C++ changes
```

To build the CPU backend (PocketFFT + OpenMP), set the backend before installing:

```bash
SKBUILD_CMAKE_DEFINE="FAST_DECONV_BACKEND=host" pip install .
```

To build wheels for distribution, use `./scripts/build.sh --python`, which writes
`dist/cuda/`, or `./scripts/build.sh all --python`, which writes one wheel per
backend to `dist/<backend>/`. Linux wheels bundle their shared dependencies,
including OpenMP and the CUDA math libraries. The NVIDIA driver and the system
C/C++ runtime must still be installed on the target machine.

For an Apptainer image with DDFacet and fast-deconv, see [APPTAINER.md](APPTAINER.md).

## Building the C++ library

`scripts/build.sh` configures and builds each backend in its own tree
(`build/<build_type>-backend_<backend>`), so CUDA and host builds never overwrite
each other.

```bash
./scripts/build.sh                # cuda, Release
./scripts/build.sh host           # host, Release
./scripts/build.sh all -t         # both backends, then run ctest
./scripts/build.sh cuda Debug     # other build types: Debug, RelWithDebInfo
./scripts/build.sh -t=Mask        # build, then ctest -R Mask
./scripts/build.sh --help         # profiling and flame-graph options
```

To build without the script, run Conan and CMake directly:

```bash
conan install . -o "fast-deconv/*:backend=cuda" -c tools.build:skip_test=False \
      --build=missing -s build_type=Release
cmake --preset conan-release-backend_cuda
cmake --build build/release-backend_cuda
```

Conan only supplies the dependencies. This recipe does not package the library.

| CMake option | Default | Effect |
|---|---|---|
| `FAST_DECONV_BACKEND` | `cuda` | `cuda` or `host` (CPU) |
| `FAST_DECONV_BUILD_TOOLS` | `OFF` | Build the `replay_ddmsc` tool (`build.sh` turns it on) |
| `BUILD_TESTING` | set by Conan | Build the GoogleTest suites |
| `FAST_DECONV_WITH_PROFILER` | `OFF` | NVTX ranges (cuda) or Tracy client (host) |
| `FAST_DECONV_SANITIZE` | `OFF` | ASan + UBSan on host code |

To leave out the tests and their dependencies, pass
`-o "fast-deconv/*:with_tests=False"` or `-c tools.build:skip_test=True` to
`conan install`. If you configure with Conan's toolchain file instead of its
preset, also pass `-DBUILD_TESTING=OFF`. Python builds never include the tests.

The build produces `libfast_deconv.a`, plus `replay_ddmsc`,
`fast_deconv_unit_tests` and `fast_deconv_nonreg_tests` when they are enabled.

## Tests

The test suite is C++ only. Run it from a build tree; for the CPU backend, use
`build/release-backend_host`.

```bash
ctest --test-dir build/release-backend_cuda -L UNIT     # kernels checked against CPU oracles
ctest --test-dir build/release-backend_cuda -L NONREG   # full synthetic DDMSC run vs baseline
```

The CUDA test binaries need a GPU to run.

The non-regression test compares the scalar metrics of a synthetic run with
`tests/baselines/ddmsc_synthetic.json`. After an intentional algorithmic change,
regenerate the baseline, review the diff, and commit it:

```bash
FAST_DECONV_UPDATE_BASELINE=1 ctest --test-dir build/release-backend_cuda -L NONREG
git diff tests/baselines/ddmsc_synthetic.json
```

## Layout

```
include/fast_deconv/      public headers: algorithm, common, core, linalg, matrix, morphology, util
  backend/<cuda|host>/    backend-specific headers
src/fast_deconv/          shared sources and the Python package
  backend/<cuda|host>/    backend implementations (.cu / .cpp)
src/python/bindings/      pybind11 wrappers
src/replay_ddmsc.cpp      dump replay tool
tests/cpp/                GoogleTest unit and non-regression tests
tests/baselines/          non-regression baseline
scripts/                  build, plotting and paper-benchmark scripts
```

## License

MIT. See [LICENSE](LICENSE).
