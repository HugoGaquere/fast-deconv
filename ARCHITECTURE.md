# Architecture

This document describes the high-level architecture of fast-deconv.

## Bird's Eye View

fast-deconv provides optimized, high-performance deconvolution for radio-interferometric imaging. It runs the
minor cycle outside the imager, on a GPU or a multi-core CPU. An imager such as DDFacet alternates major cycles,
which grid visibilities into a dirty image, with minor cycles, which find the sky components in that image.
fast-deconv implements one minor-cycle algorithm, DDMSC: multi-scale, multi-frequency CLEAN with
direction-dependent (per-facet) PSFs.

- **Fixed for an imaging run:**
  - per-facet, per-channel PSFs, `(n_facets, n_freq, psf_h, psf_w)`;
  - a spectral basis, `(n_freq, n_order)`;
  - a pixel-to-facet map, `(nrow, ncol)`;
  - an external mask, `(nrow, ncol)`;
  - the scales: a Gaussian sigma and a bias for each, `(n_scales,)` each.
- **Handed over by each major cycle:**
  - a residual cube, `(n_freq, nrow, ncol)`;
  - a Jones normalization, `(n_freq, nrow, ncol)`;
  - per-channel weights, `(n_freq,)`.
- **Each minor cycle repeatedly:**
  - picks the best scale;
  - cleans the components at that scale;
  - fits each component's spectrum.
- **Output:**
  - the residual, overwritten in place, `(n_freq, nrow, ncol)`;
  - the components: positions `(n_components, 2)`, scales and gains `(n_components,)` each, spectral coefficients
    `(n_components, n_order)`;
  - the reason it stopped, one `convergence_status`.

fast-deconv is a library. The imager stays in Python and calls a `DDMSC` object: one session per
imaging run, with one `run()` per major cycle. The session keeps the device, the staged PSFs and the component
history across calls.

```
DDFacet                                     fast-deconv
DDMSC(raw_psfs, xdes, mask, …)         ──►  stage inputs, create the pool and both contexts
for each major cycle:
    grid → residual cube
    run(dirty, jones_norm, w)          ──►  minor cycle; dirty ← new residual
                                       ◄──  DDMSCResult
    predict and subtract components
```

The same code runs on two backends, chosen at configure time by `FAST_DECONV_BACKEND`: `host` (OpenMP on multi-core
CPUs, the default) and `cuda` (NVIDIA GPUs). A build contains exactly one backend.

## Code Map

The C++ library is layered. Each layer includes only the layers below it:

```
                                                                    fd_backend (cuda | host)
python       DDMSC, DDMSCResult
   │
algorithm    Ddmsc
   │
common       subtract_component, fit_coefficients, compute_gain,
   │         build_auto_mask, convergence, select_best_scale,
   │         psf_convolution
   │
linalg       gaussian_convolution_ctx, weighted_sum      ◄──── gaussian_convolution_ctx, fft
matrix       find_peak, tiled_argmax_ctx, max_and_rms    ◄──── tiled_argmax_impl, stats_types
morphology   binary_dilation, compute_mask_roi           ◄──── binary_dilation (source only)
   │
core         exec_resources, exec_ctx, span*/cont*,      ◄──── exec_impl, memory_types,
             index2d, narrow, FD_LOG_*, FD_PROFILE_*           for_each_index, reduce, launch
```

`linalg`, `matrix` and `morphology` sit side by side, and `util` sits beside `core`. `fd_backend` is not a layer of
its own: `core`, `linalg` and `matrix` take backend-specific headers from it, `morphology` takes a backend-specific
source, and the generic sources in every layer call its `for_each_index` and `reduce`.

Public headers live under `include/fast_deconv/<layer>/`, and the sources that implement them live under
`src/fast_deconv/<layer>/`:

```
fast-deconv/
├── include/fast_deconv/          public headers
│   ├── core/                     execution model, array types, logging, profiling
│   ├── linalg/                   Gaussian convolution, FFT sizing, weighted_sum
│   ├── matrix/                   peak search (full and tiled), max and standard deviation
│   ├── morphology/               binary dilation, mask bounding box
│   ├── common/                   deconvolution building blocks: subtraction, spectral fit, masks, scales
│   │   └── detail/               spectral fit math, shared by host and device
│   ├── algorithm/                the deconvolution algorithms, one header/source pair each (ddmsc)
│   ├── util/                     CUDA/cuFFT error macros, .npy dump
│   └── backend/
│       ├── cuda/fd_backend/      on the include path when FAST_DECONV_BACKEND=cuda
│       │   ├── core/             exec_impl, for_each, reduce, memory_types, launch.cuh
│       │   ├── linalg/           fft, gaussian_convolution
│       │   └── matrix/           tiled_argmax_impl, stats_types
│       └── host/fd_backend/      on the include path when FAST_DECONV_BACKEND=host; same files but launch.cuh
├── src/
│   ├── fast_deconv/
│   │   ├── linalg/ … algorithm/  implementations, one directory per layer (core is header-only)
│   │   ├── backend/cuda/         .cu kernels, cuFFT
│   │   ├── backend/host/         OpenMP loops, DUCC
│   │   ├── __init__.py           Python package, FAST_DECONV_DUMP capture
│   │   └── *.pyi, py.typed       type stubs
│   ├── python/                   pybind11 module and bindings
│   ├── replay_ddmsc.cpp          replay tool for captured sessions
│   └── npy_loader.hpp            .npy reader for the replay tool
├── tests/
│   ├── cpp/                      GoogleTest: UNIT and NONREG
│   └── baselines/                non-regression baseline JSON
├── scripts/                      build, verification, plots, paper benchmarks
└── cmake/                        Conan provider, wheel runtime bundling
```

### `include/fast_deconv/core`

The execution model, which every other layer is written against.

`exec_resources` owns what the backend runs on and allocates from: on CUDA, the device and a stream-ordered memory
pool; on host, a `block_pool`. `exec_ctx` is one execution context created from it. On CUDA, a context is one
stream. It allocates (`alloc_mdcontainer`, `alloc_ptr`), copies, zeroes, waits, and moves caller memory to and from
the backend (`stage`, `unstage`).

`memory_types.hpp` defines the array vocabulary: `span1d..span4d` (non-owning views) and `cont1d..cont4d` (owning,
refcounted containers) in backend memory, plus `host_span*`/`host_cont*` for caller memory. All are emu mdspans with
`int64` extents and row-major layout. `dims.hpp` has `narrow`, `ceil_div` and `count`. `index.hpp` has `index2d` and
`unravel_index_2D`.

`logger.hpp` (`FD_LOG_*`, over spdlog) and `profiler.hpp` (`FD_PROFILE_*`, over NVTX or Tracy) are the only ways the
code logs and profiles.

**Architecture Invariant:** arrays cross function boundaries as `core::` spans and containers, never as a raw
pointer plus separate extents. The one exception is the FFT scratch in `fd_backend/linalg/fft.hpp`, which is opaque
and sized entirely by `fft_dims`.

**Architecture Invariant:** counts and indices are `std::int64_t`. A narrower type, such as an argument to cuFFT, a
launch dimension or a `size_t` shape, is reached only through `core::narrow`, which throws on overflow.

### `include/fast_deconv/backend/<cuda|host>/fd_backend` and `src/fast_deconv/backend/<cuda|host>`

The two backends. CMake puts exactly one of the `backend/<name>` include directories on the include path, so
`#include <fd_backend/...>` resolves to the selected backend. The backend's `.cu`/`.cpp` files under
`src/fast_deconv/backend/<name>/` are picked up by a glob. Each backend provides the same set of files:

| `fd_backend` file | provides | cuda | host |
|---|---|---|---|
| `core/exec_impl.hpp` | `exec_resources_impl`, `exec_ctx_impl`, which `core::exec_ctx` derives from | device and `cudaMemPool`; one stream per context | `block_pool`; work runs inline |
| `core/memory_types.hpp` | what `span*`/`cont*` mean in this build | device views | the host views themselves |
| `core/for_each.hpp` | `for_each_index` over 1D, 2D and 3D index spaces | kernel, one thread per index | OpenMP loop; serial below `kMinParallelForEach` |
| `core/reduce.hpp` | `reduce(ctx, n, map, combine, identity)` | CUB `DeviceReduce` | OpenMP per-thread partials |
| `linalg/gaussian_convolution.hpp` | `gaussian_convolution_ctx` | padded 2D FFT, cuFFT | separable rows then columns, DUCC |
| `matrix/tiled_argmax_impl.hpp` | the tile cache behind `tiled_argmax_ctx`, checked by the `tiled_argmax_backend` concept | device cache, one block per tile | host cache |
| `matrix/stats_types.hpp`, `linalg/fft.hpp` | small backend-specific types | | |

`binary_dilation` has no `fd_backend` header: `morphology/dilation.hpp` declares it, and each backend defines it in
`src/fast_deconv/backend/<name>/morphology/`.

On the host backend, `core::exec_ctx::stage()` borrows the caller's memory, because backend memory is host
memory. On CUDA it copies.

`backend/cuda/fd_backend/core/launch.cuh` has `launch_cfg` and `launch`, which every hand-written kernel launch
goes through.

**Architecture Invariant:** only the backend directories (and the CUDA-only `util/` helpers) mention CUDA, cuFFT,
CUB, Thrust, OpenMP or DUCC. Nothing outside them contains a `__global__` function or a `#pragma omp`.

### "Generic" sources

Some sources under `src/fast_deconv/` are written once, against `for_each_index` and `reduce`, with
`FD_HOST_DEVICE` lambdas. They are listed in `FAST_DECONV_GENERIC_SOURCES` in `CMakeLists.txt`: `matrix/argmax`,
`matrix/stats`, `matrix/tiled_argmax`, `morphology/roi`, `common/subtract`, `common/mask`, `common/multi_frequency`
and `linalg/linalg`. In a CUDA build, nvcc compiles them so the lambdas become device code. The other
sources are plain C++ in both builds.

If a new source includes `fd_backend/core/for_each.hpp` or `reduce.hpp`, add it to that list. The CUDA headers
`#error` out otherwise.

**Architecture Invariant:** an algorithm is written once. Code that differs between backends lives in `fd_backend` or
the backend sources, behind an interface with the same name and signature in both backends, and is chosen by the include
path, not by `#ifdef`.

### `linalg`

`gaussian_convolution_ctx` (forward once, then `convolve` per sigma) and the arithmetic that sizes it:
`gaussian_reach`, `max_gaussian_reach`, `fft_dims` and `next_fast_size`. Also `weighted_sum`, which collapses the
channel axis into the mean residual.

**Architecture Invariant:** convolutions are linear, not circular. The padding comes from `max_gaussian_reach`, the
distance at which the kernel falls below float32 precision, so nothing wraps around.

### `matrix`

Peak finding and image statistics. `peak`, `peak_max` and `peak_criterion` (mask plus optional absolute value)
define what "the peak" means. `find_peak` scans the full image. `tiled_argmax_ctx` caches one peak per tile, so
after a subtraction it rescans only the tiles under the PSF footprint (`run_incremental`). `max_and_rms` does a
fused max and standard-deviation reduction.

**Architecture Invariant:** ties go to the smaller flat index, in every backend and every path (`peak_max`,
`kPeakIdentity`). This keeps the full scan, the tiled scan and the two backends picking the same pixel.

### `morphology`

`binary_dilation` and `compute_mask_roi` (which returns a `roi`), used only to build the auto-mask.

### `common`

The building blocks the algorithms are made of, each usable and tested on its own:

- `subtract.hpp`: `subtract_component`, for a single plane and per channel.
- `multi_frequency.hpp`: `fit_coefficients`, the per-component spectral fit. The math is in
  `detail/spectral_fit.hpp`, written once for host and device.
- `gain.hpp`: `compute_gain`, the CLEAN gain of a scale-convolved PSF.
- `mask.hpp`: `build_auto_mask`, the per-scale masks around the components found so far.
- `convergence.hpp`: `convergence` and `scale_stall_tracker`, the stopping logic. This file is pure host C++.
- `scales.hpp`: `select_best_scale`, which convolves the mean residual at each live scale and keeps the scale with
  the highest biased peak.
- `psf_convolution.hpp`: a lazy cache of scale-convolved PSFs for each (scale, facet), plus their gains. It holds
  only the current scale's entries.

Namespaces do not always match directories: `common/multi_frequency.hpp` is `fast_deconv::multi_frequency`, and
`common/scales.hpp` is `fast_deconv::scale`.

### `algorithm`

The deconvolution algorithms, and nothing else: each is one `<name>.hpp`/`<name>.cpp` pair, and anything another
algorithm could reuse goes to `common`. Today there is one, DDMSC:

- `ddmsc.hpp`: `params` (with the defaults), `context`, `ddmsc_result` and `Ddmsc`, the session object exposed to
  Python. `Ddmsc` holds a public `ctx` and `cfg`; `run()` stages the per-call inputs, runs the cycle loop with
  `cfg`, then unstages the residual. The bindings expose each `cfg` field as a property;
  a few Python names differ, e.g. `max_sub_iteration` for `max_clean_iteration`.
- `ddmsc.cpp`: the session and the loop, shown below.

The cycle loop (`run_ddmsc_cycles`, private to `ddmsc.cpp`) nests two loops. The outer loop picks a scale; the inner
clean loop extracts components at it:

```
weighted_sum → mean residual; max_and_rms → stop_flux
└─ outer loop, until convergence stops
   ├─ auto-mask engaged? build mask_per_scale (once)
   ├─ auto-mask on? combine_scale_masks over the live scales
   ├─ select_best_scale → scale s, scaled residual, first peak
   ├─ clean loop, while peak > peak_factor · first peak and under max_clean_iteration
   │  ├─ psf_cache.get(s, facet of peak)
   │  ├─ fit_coefficients              cube_exec
   │  ├─ subtract_component (cube)     cube_exec
   │  ├─ subtract_component (scaled)   exec
   │  └─ tiled argmax, incremental     exec
   ├─ wait both
   ├─ no component found? stop
   └─ weighted_sum, max_and_rms, convergence.track
```

`context` holds two `exec_ctx`s. `exec` runs the scale search and the clean loop on the scaled residual.
`cube_exec` runs the per-channel fit and subtraction on the full cube, overlapping with the first. They meet at
explicit `wait()`s at the end of each outer iteration. Within the clean loop, the incremental argmax syncs `exec`
each iteration, and the host enqueues the next fit on `cube_exec` only once that peak is back:

```
exec ─────────── select scale ─┬─ sub scaled ─ argmax ─┬─ sub scaled ─ argmax ─ … ─┬─ weighted_sum, stats
cube_exec ─────────────────────┴─ fit ─ sub cube ──────┴─ fit ─ sub cube ──── … ───┘
                                                                         wait both ┘
```

**Architecture Invariant:** `algorithm/` is plain C++ even in a CUDA build. It contains no kernels and no
`for_each_index`; it orchestrates calls to the layers below.

### `src/python` and `src/fast_deconv`

`src/python/bindings/ddmsc.cpp` binds `Ddmsc`, `ddmsc_result` and `convergence_status` with pybind11. NumPy arrays
become `core::host_span*` through emu's casters, without a copy. `keep_alive` keeps borrowed arrays alive as long as
the session. `src/fast_deconv/` is the Python package: `__init__.py`, the `.pyi` stubs and `py.typed`.
When `FAST_DECONV_DUMP` is set, `__init__.py` swaps in a `DDMSC` subclass that writes every constructor and `run()`
input to `.npy` files.

**Architecture Invariant:** only `src/python` knows about Python, and the bindings use only the backend-neutral
public API. The extension is compiled as plain C++ for both backends.

### `src/replay_ddmsc.cpp` and `src/npy_loader.hpp`

`replay_ddmsc` reads a `FAST_DECONV_DUMP` directory and replays the session in C++, without DDFacet. It can
override parameters, crop the image, repeat runs, and run a fixed-work `--bench` mode. This is the entry point for
debugging, profiling (nsys, ncu, Tracy) and benchmarking on real data. It is built when
`FAST_DECONV_BUILD_TOOLS=ON`.

### `include/fast_deconv/util`

`CHECK_CUDA` and `REPORT_CUDA` (`cuda_macros.hpp`), `CUFFT_CALL` and `REPORT_CUFFT` (`cufft_macros.hpp`), `dump_npy`
for debugging (`dump.hpp`). Everything here is CUDA-only.

### `tests`

`tests/cpp` has two GoogleTest binaries. `fast_deconv_unit_tests` (ctest label `UNIT`) checks primitives against host
oracles in `helpers/host_oracles.hpp`, plus the DDMSC auto-mask. `fast_deconv_nonreg_tests` (label `NONREG`) runs a full
synthetic DDMSC session and compares its scalar metrics with `tests/baselines/ddmsc_synthetic.json`.

### `scripts`

`build.sh` builds each backend in its own tree, `build/<build_type>-backend_<backend>`, running Conan and the CMake
configure only on a fresh tree (`--clean` starts over). `verify_all.sh` runs the full pipeline: format, clang-tidy,
build, tests, compute-sanitizer and ASan/UBSan. Benchmark, plotting and paper scripts live outside the repo, in
`../fast-deconv-scripts`.
Nothing in `scripts/` is part of the library.

## API Boundaries

**API Boundary:** the Python `DDMSC` class. Its shapes, dtypes, property names and `DDMSCResult` fields are
documented in `README.md` and typed in `src/fast_deconv/_fast_deconv/ddmsc.pyi`. Changing them breaks DDFacet.

**API Boundary:** `algorithm::ddmsc::Ddmsc`, the C++ form of the same contract, used by `replay_ddmsc` and the
non-regression test. It takes host views only. Staging into backend memory is its job, not the caller's. Where
each input lives depends on who reads it:

| input | cuda | host | must outlive | reason |
|---|---|---|---|---|
| `raw_psfs`, `xdes`, `mask` | copied to the device once | borrowed | the session, on host | read by kernels: PSF convolution, spectral fit, peak criterion |
| `scale_sigmas` | copied to a `std::vector` | copied to a `std::vector` | nothing | read on the host: each sigma is passed by value to a convolution |
| `scale_bias` | borrowed | borrowed | the session | read on the host: multiplies each scale's peak after the search returns |
| `map_pixel_facet` | borrowed | borrowed | the session | read on the host: the facet of each peak picks the PSF cache entry |
| `dirty` | copied to the device, residual copied back | written in place | the call | read and written by kernels; the caller gets the residual |
| `jones_norm`, `weights_freq` | copied to the device per call | borrowed | the call | read by kernels: spectral fit, `weighted_sum`, gains |

On the Python side, `keep_alive` ties every constructor array that may be borrowed to the session.

Everything below `Ddmsc` is internal. Headers under `include/` are public only so that tests and tools can
include them.

**API Boundary:** the `fd_backend` file set and the backend sources. Adding a backend means providing every `fd_backend`
file listed above and `binary_dilation`, with the same names and signatures. `tiled_argmax_backend` checks one of them
at compile time.

## Cross-Cutting Concerns

### Memory and streams

All working memory comes from the context's pool through `exec_ctx::alloc*`. There is no `cudaMalloc`, `new[]` or
`std::vector` of device data. Pools never return memory to the system, because every minor cycle repeats the same
allocation sizes. On CUDA, allocation and freeing are stream-ordered: memory freed on one stream must not still be
in use by another. This is why `psf_convolution::get` says to synchronize before switching scales.

Calls that bring a value back to the host synchronize their context; their docs say "Syncs @p ctx". Everything
else is only enqueued.

On the host backend, `stage()` borrows instead of copying. The table under API Boundaries says which caller memory
must outlive what.

### Error handling

Errors are exceptions. CUDA and cuFFT calls go through `CHECK_CUDA`/`CUFFT_CALL`, and kernel launches through
`core::launch`, which reports the call site. Destructors and `noexcept` paths use `REPORT_CUDA`/`REPORT_CUFFT`,
which only print the error. The library never exits the process. Non-finite input residuals are rejected at the
start of `run_ddmsc_cycles`. A singular spectral fit writes NaN, which reaches the residual; `convergence` reports a
non-finite peak or standard deviation as `diverged`.

### Testing

Tests are written once against `core::exec_ctx` and run unchanged on both backends. Host oracles accumulate in
double. The non-regression baseline is shared by both backends: it stores values only, and the per-metric
tolerances live in `test_ddmsc_nonreg.cpp` so that regenerating the baseline cannot silently loosen them.
Regenerate it with `FAST_DECONV_UPDATE_BASELINE=1` and review the diff.

### Observability

Logging goes through `FD_LOG_*`. `FD_LOG_DEBUG` and `FD_LOG_TRACE` are compiled out of Release builds. The
`run_ddmsc` banner logs every parameter of a run at INFO level, and a recap at the end logs the time, component
count, exit status, flux and standard deviation before and after, and pool usage.

Profiling goes through `FD_PROFILE_*` and is compiled out unless `FAST_DECONV_WITH_PROFILER=ON`. With profiling on,
CUDA builds emit NVTX ranges and host builds emit Tracy zones, plots and allocation events.

To reproduce a problem on real data, capture it with `FAST_DECONV_DUMP` and replay it with `replay_ddmsc`.

### Build and packaging

CMake drives the build. Conan supplies the dependencies, through `cmake/conan_provider.cmake` for Python builds and an
explicit `conan install` in `build.sh`; emu comes from the `cosmic` remote. DUCC
(not on ConanCenter) and Tracy (whose client must match the profiler's version exactly) come from `FetchContent`. Python
wheels are built with scikit-build-core, which reads `FAST_DECONV_BACKEND` from the environment and keeps one build
directory per backend, since the two enable different CMake languages. `cmake/install_runtime_dependencies.cmake`
bundles the shared libraries next to the module. The library is static and is linked into the Python extension.
