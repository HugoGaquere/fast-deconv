# fast-deconv

High-performance DDMSC deconvolution for radio astronomy, on GPUs and multi-core CPUs, callable from Python.

fast-deconv is an optimized multi-scale and multi-frequency deconvolution engine with two execution backends, both tuned for speed: CUDA for NVIDIA GPUs and OpenMP for multi-core CPUs.
It's designed for integration with [DDFacet](https://github.com/cyriltasse/DDFacet)-style imaging pipelines.

> **Status:** beta (`v0.9.2`). The API may still change between minor versions.

## Requirements

- Linux
- A C++20 compiler with OpenMP
- Python ≥ 3.10; NumPy is the only runtime dependency
- **CUDA backend only:** the CUDA Toolkit and an NVIDIA GPU

CMake and Conan come from `pip` in the setup below. The C++ libraries (emu, fmt, spdlog,
DUCC, ...) are fetched by the build, so there is nothing else to install.

## First-time setup

Do this once per machine.

### 1. Virtual environment and build tools

Create and activate a venv at the repo root, then install the build tools: `conan`, `cmake` and `ninja`.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install conan cmake ninja
```

[uv](https://docs.astral.sh/uv/) works too and is faster: run `uv venv` instead of
`python3 -m venv .venv`, and `uv pip` wherever this README says `pip`.

### 2. Conan profile

A Conan profile describes your toolchain (OS, architecture, compiler and its version, C++
standard, build type). Conan uses it to choose or build dependency binaries that match your build.
`detect` creates one from the compiler it finds:

```bash
conan profile detect --exist-ok
conan profile path default   # the file to edit
```

In that file, set `compiler.cppstd=gnu20` to match the project. `pip install` derives its own
profile from CMake, but `scripts/build.sh` and plain `conan install` use this one.

Optional defaults, in `$(conan config home)/global.conf`:

- `tools.cmake.cmaketoolchain:generator=Ninja`: use Ninja instead of Make.
- `tools.build:skip_test=True`: don't fetch or build GoogleTest unless a command asks for it.
  `scripts/build.sh -t` overrides it.

### 3. The `cosmic` remote

fast-deconv depends on [emu](https://gitlab.obspm.fr/cosmic/tools/emu), which is not on
ConanCenter. Add the remote that hosts it, which allows anonymous reads, then check that emu
resolves:

```bash
conan remote add cosmic https://gitlab.obspm.fr/api/v4/projects/5071/packages/conan
conan list "emu/*" -r cosmic   # should print a list of emu versions
```

### 4. CUDA Toolkit (CUDA backend only)

Skip this step if you only build the CPU backend.

Conan and CMake locate CUDA with CMake's `FindCUDAToolkit`, so `nvcc` must be on `PATH`, or
`CUDAToolkit_ROOT` must point at the toolkit:

```bash
export PATH=/usr/local/cuda/bin:$PATH
nvcc --version
```

CUDA builds target the GPU in the build machine (`native`) by default. To build for
other GPUs, or on a machine without one, set `CMAKE_CUDA_ARCHITECTURES` or
`CUDAARCHS`, e.g. `CUDAARCHS="80;90"`.

## Quick start

Build and install the package into the active environment, like any pip package. Choose a
backend; each build contains exactly one. To work on fast-deconv itself, see
[Development](#development) instead.

```bash
pip install .                           # CPU: OpenMP, no GPU needed
FAST_DECONV_BACKEND=cuda pip install .  # CUDA: NVIDIA GPU + CUDA Toolkit
```

The first install compiles the dependencies that have no prebuilt binary for your profile,
including emu's Boost, so it takes a while. Later builds reuse Conan's cache in `~/.conan2`.

The Python API is the same for both backends.

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
    mask=np.zeros((n, n), bool),                            # True = masked out
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
| `mask` | `(nrow, ncol)` | `bool` | External mask, `True` = pixel excluded from cleaning |
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

`release()` frees every buffer the session holds on the GPU (on host, its pooled RAM) so that the
next step, such as GPU gridding, can use it. The parameters and the component history the auto-mask
is built from are kept, and the next `run()` uploads the PSFs, `xdes` and `mask` again. Call it only
when something else needs the memory: on host, it gives up the pool reuse between cycles.

Types are in `src/fast_deconv/_fast_deconv/ddmsc.pyi`.

## Development

### Editable Python install

Unlike `pip install .`, an editable install rebuilds the extension on import after C++
changes:

```bash
uv sync                           # host
FAST_DECONV_BACKEND=cuda uv sync  # cuda
```

### Wheels for distribution

To build wheels to install on other machines, use `./scripts/build.sh --python`, which writes
`dist/host/`, or `./scripts/build.sh all --python`, which writes one wheel per
backend to `dist/<backend>/`. Linux wheels bundle their shared dependencies,
including OpenMP and the CUDA math libraries. The NVIDIA driver and the system
C/C++ runtime must still be installed on the target machine.

### Building the C++ library

`scripts/build.sh` configures and builds each backend in its own tree
(`build/<build_type>-backend_<backend>`), so CUDA and host builds never overwrite
each other.

```bash
./scripts/build.sh                # host, Release
./scripts/build.sh cuda           # cuda, Release
./scripts/build.sh all -t         # both backends, then run ctest
./scripts/build.sh cuda Debug     # other build types: Debug, RelWithDebInfo
./scripts/build.sh -t=Mask        # build, then ctest -R Mask
./scripts/build.sh cuda --clean   # delete the tree, then rebuild from conan install
./scripts/build.sh --help         # profiling and flame-graph options
```

Conan and CMake configure run only on a fresh tree; later runs just build. Pass `--clean`
after changing `conanfile.py`, the Conan profile, `-t`, `--profile` or `--flamegraph`.

To build without the script, run Conan and CMake directly:

```bash
conan install . -o "fast-deconv/*:backend=host" -c tools.build:skip_test=False \
      --build=missing -s build_type=Release
cmake --preset conan-release-backend_host
cmake --build build/release-backend_host
```

For the CUDA backend, replace `host` with `cuda` in all three commands.

Conan only supplies the dependencies. This recipe does not package the library.

| CMake option | Default | Effect |
|---|---|---|
| `FAST_DECONV_BACKEND` | `host` | `cuda` or `host` (CPU) |
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

### Tests

The test suite is C++ only. Run it from a build tree built with `-t`:

```bash
ctest --test-dir build/release-backend_host -L UNIT     # kernels checked against CPU oracles
ctest --test-dir build/release-backend_host -L NONREG   # replay of a captured DDMSC session
```

For the CUDA backend, use `build/release-backend_cuda` instead. Its test binaries need a GPU to run.

The non-regression test replays the DDMSC session dumped in `tests/data/lofar_lba_point_sources/`, cycle by
cycle, and compares each cycle with the expected values in `tests/cpp/test_ddmsc_nonreg.cpp`: exit status,
outer iterations, components per scale, final peak and rms, and the sum of the order-0 coefficients. The
session is a simulated LOFAR LBA field of 9 point sources with direction-dependent gains: 975 × 975 px,
6 bands, 9 facets, 4 scales.

After an intentional algorithmic change, run the test: it prints one row per cycle in the format of
`kExpected`. Paste the rows, review the diff, and commit it.

## Architecture

For the code layout, the layers, the backends and their invariants, see [ARCHITECTURE.md](ARCHITECTURE.md).

## Contributing

TODO

## Third-party libraries

fast-deconv builds on:

- [emu](https://gitlab.obspm.fr/cosmic/tools/emu)
- [fmt](https://github.com/fmtlib/fmt)
- [spdlog](https://github.com/gabime/spdlog)
- [pybind11](https://github.com/pybind/pybind11)
- [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit)
- [DUCC](https://gitlab.mpcdf.mpg.de/mtr/ducc)
- [Tracy](https://github.com/wolfpld/tracy)
- [GoogleTest](https://github.com/google/googletest)

## License

MIT. See [LICENSE](LICENSE).
