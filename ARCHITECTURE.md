# Architecture

This document describes the high-level architecture of fast-deconv and the rules its code follows.

## Bird's Eye View

fast-deconv runs the deconvolution minor cycle of radio-interferometric imaging outside the imager, on a GPU or a
multi-core CPU. An imager such as DDFacet alternates major cycles, which grid visibilities into a dirty image, with
minor cycles, which find the sky components in that image. fast-deconv implements one minor-cycle algorithm, DDMSC:
multi-scale, multi-frequency CLEAN with direction-dependent (per-facet) PSFs.

fast-deconv is a library. The imager stays in Python and drives a `DDMSC` session: one per imaging run, one `run()`
per major cycle. The session keeps the device, the staged PSFs and the component history across calls.

The same code runs on two backends, chosen at configure time by `FAST_DECONV_BACKEND`: `host` (OpenMP, the default)
and `cuda`. A build contains exactly one backend. Both are performance targets; neither is a fallback.

```
DDFacet                                     fast-deconv
DDMSC(raw_psfs, xdes, mask, …)         ──►  stage inputs, create the pool and both contexts
for each major cycle:
    grid → residual cube
    run(dirty, jones_norm, w)          ──►  minor cycle; dirty ← new residual
                                       ◄──  DDMSCResult
    predict and subtract components
```

The session runs on two execution contexts that overlap: `exec` works on the scaled residual (scale search, clean
loop, peak search), and `cube_exec` on the full cube (per-channel fit and subtraction).

`run_ddmsc_cycles` in `src/fast_deconv/algorithm/ddmsc.cpp` sketches one `run()` step by step.

## Code Map

The C++ library is layered. Each layer includes only the layers below it:

```
python       DDMSC, DDMSCResult                         bindings, plain C++
algorithm    Ddmsc                                      orchestration only
common       subtract, multi_frequency, gain, mask,     deconvolution building blocks
             convergence, scales, psf_convolution
linalg │ matrix │ morphology                           convolution, peak search and stats, dilation
core                                                    execution model, array types, logging, profiling
util                                                    CUDA helpers, beside core
fd_backend   (cuda | host)                              what differs between backends, under every layer
```

```
fast-deconv/
├── include/fast_deconv/
│   ├── <layer>/                      public headers, one directory per layer
│   └── backend/<cuda|host>/fd_backend/   backend headers; one is on the include path
├── src/fast_deconv/
│   ├── <layer>/                      implementations (core is header-only)
│   ├── backend/<cuda|host>/          backend sources: .cu kernels and cuFFT, or OpenMP and DUCC
│   └── __init__.py, *.pyi            Python package and stubs
├── src/python/                       pybind11 bindings
├── src/replay_ddmsc.cpp              replays a FAST_DECONV_DUMP session in C++
├── tests/cpp/                        GoogleTest: UNIT and NONREG
├── tests/baselines/                  non-regression baseline
├── scripts/                          build and verification, not part of the library
└── cmake/                            Conan provider, wheel runtime bundling
```

- `core`: `exec_resources` (device and pool) and `exec_ctx` (one stream, its allocations and copies); the array
  vocabulary (`span*`, `cont*`, `host_span*`, `host_cont*`); `narrow`, `index2d`; `FD_LOG_*`, `FD_PROFILE_*`.
- `linalg`: `gaussian_convolution_ctx`, its FFT sizing, `weighted_sum`.
- `matrix`: `find_peak`, `tiled_argmax_ctx`, `max_and_rms`.
- `morphology`: `binary_dilation`, `compute_mask_roi`, used only by the auto-mask.
- `common`: the pieces algorithms are built from, each usable and tested on its own.
- `algorithm`: one `<name>.hpp`/`<name>.cpp` pair per algorithm. Today only `ddmsc`.
- `util`: CUDA-only helpers: `CHECK_CUDA`, `CUFFT_CALL`, `dump_npy`.

A layer's code lives in `fast_deconv::<layer>`; each algorithm gets its own, `fast_deconv::algorithm::<name>`.

## Rules

The patterns to follow when adding code.

### Every operation takes an `exec_ctx` first

A function that does work takes `const core::exec_ctx&` as its first parameter. The context decides where the work
runs and where its memory comes from:

```cpp
void subtract_component(const core::exec_ctx& ctx, core::span2d<float> residual, core::span2d<const float> psf, …);
stats_result max_and_rms(const core::exec_ctx& ctx, core::span2d<const float> data, …);
```

On CUDA, work is only enqueued; on host, it runs inline. A call that brings a value back to the host synchronizes
its context, and its doc says "Syncs @p ctx".

### Stateful operations are objects

An operation that builds state once and is called many times is a class. It takes the context in its constructor and
keeps a `const core::exec_ctx&` to it, so the context must outlive it:

```cpp
linalg::gaussian_convolution_ctx conv{ctx, batch, nrow, ncol, sigmas};
conv.forward(input, spectrum);              // once
conv.convolve(spectrum, sigma, out);        // per sigma

matrix::tiled_argmax_ctx tiled{ctx, extents, 64};
auto first = tiled.run(residual);
auto next = tiled.run_incremental(residual, row, col, fh, fw);
```

### Arrays are `core::` spans and containers

Arrays cross function boundaries as `core::span1d..span4d` (views) or `cont1d..cont4d` (owning, refcounted), with
`const T` for inputs. Never a raw pointer plus separate extents. Caller memory is `host_span*`/`host_cont*`; only
`exec_ctx::stage()`/`unstage()` cross between the two. The one exception is the opaque FFT scratch in
`fd_backend/linalg/fft.hpp`, sized by `fft_dims`.

### Counts and indices are `std::int64_t`

A narrower type (a cuFFT argument, a launch dimension, a `size_t` shape) is reached only through `core::narrow`,
which throws on overflow.

### Memory comes from the context

All working memory comes from `exec_ctx::alloc_mdcontainer` or `alloc_ptr`, and is freed through the context. No
`cudaMalloc`, `new[]` or `std::vector` of device data. Pools give memory back only on `Ddmsc::release()`, since every
minor cycle repeats the same sizes. On CUDA, allocation is stream-ordered: memory freed on one context must not still be in use on
another.

### Backend differences live in `fd_backend`, chosen by the include path

CMake puts exactly one `include/fast_deconv/backend/<name>` on the include path, so `#include <fd_backend/...>`
resolves to the selected backend. No `#ifdef` selects a backend. Code that differs takes one of three forms:

- **Same class, both backends.** The backend header defines the whole class, with the same name and signature in
  each. The public header includes it last. Example: `gaussian_convolution_ctx`.
- **Generic class over a backend `*_impl`.** The public class is written once and holds or derives from an
  `fd_backend` `*_impl`. When the `*_impl` has a contract of its own, a concept states it and a `static_assert`
  checks it:

  ```cpp
  #include <fd_backend/matrix/tiled_argmax_impl.hpp>

  template <typename T>
  concept tiled_argmax_backend =
      std::constructible_from<T, const core::exec_ctx&, const tile_grid&> &&
      requires(T impl, core::span2d<const float> data, peak_criterion criterion, tile_range range) {
        { impl.reduce_tiles(data, criterion, range) } -> std::same_as<void>;
        { std::as_const(impl).tiles() } -> std::same_as<const peak*>;
      };
  static_assert(tiled_argmax_backend<tiled_argmax_impl>);

  class tiled_argmax_ctx { … tiled_argmax_impl impl_; };
  ```

  `core::exec_ctx` derives from `exec_ctx_impl` the same way.
- **Declared once, defined per backend.** The public header declares the function, and each backend source under
  `src/fast_deconv/backend/<name>/` defines it. Example: `binary_dilation`.

Only the backend directories (and the CUDA-only `util/`) mention CUDA, cuFFT, CUB, Thrust, OpenMP or DUCC. Nothing
outside them has a `__global__` function or a `#pragma omp`.

### Simple kernels are written once, as generic sources

A loop or a reduction is written once against `core::for_each_index` and `core::reduce`, with an `FD_HOST_DEVICE`
lambda. The CUDA backend runs it as a kernel, the host backend as an OpenMP loop:

```cpp
core::for_each_index(ctx, nrow, ncol, [=] FD_HOST_DEVICE(std::int64_t r, std::int64_t c) { … });
```

A source that includes `fd_backend/core/for_each.hpp` or `reduce.hpp` must be listed in
`FAST_DECONV_GENERIC_SOURCES` in `CMakeLists.txt`, so nvcc compiles it in a CUDA build. The CUDA headers `#error`
otherwise. Hand-written CUDA kernels go through `core::launch`.

### `algorithm/` only orchestrates

An algorithm is plain C++ in every build: no kernels, no `for_each_index`. It calls the layers below. Anything
another algorithm could reuse belongs in `common`.

### Errors are exceptions

CUDA and cuFFT calls go through `CHECK_CUDA`/`CUFFT_CALL`. Destructors and `noexcept` paths use
`REPORT_CUDA`/`REPORT_CUFFT`, which only print. The library never exits the process. Invalid arguments throw
`std::invalid_argument`. Numerical failure is not an exception: a NaN reaches the residual and `convergence` reports
`diverged`.

### Logging and profiling go through the macros

`FD_LOG_*` (spdlog; `DEBUG` and `TRACE` compiled out of Release) and `FD_PROFILE_*` (NVTX on CUDA, Tracy on host;
compiled out unless `FAST_DECONV_WITH_PROFILER=ON`). A function that enqueues work starts with `FD_PROFILE_FN()`.

### Naming

- The execution context parameter is `ctx`, and a class stores it as `ctx_`.
- Private members and private member functions end in `_`.
- `*_impl`: the backend half of a generic class.
- Identifiers are `snake_case`; constants are `kCamelCase`.

### Comments and docs

- Public declarations carry Doxygen: `@brief`, one `@param[in|out]` line per parameter, `@return`, `@throws`,
  `@pre`. A `@param` fits on one line; anything longer goes in the body.
- A function that syncs its context says "Syncs @p ctx". Device memory is marked in the parameter doc ("device").
- Comments in the code are one line and state a constraint or a reason, not what the next line does.

### Tests are written once

Tests take a `core::exec_ctx` and run unchanged on both backends, checked against double-precision oracles in
`tests/cpp/helpers/host_oracles.hpp`. The non-regression baseline stores values only; its tolerances live in
`test_ddmsc_nonreg.cpp`, so regenerating it (`FAST_DECONV_UPDATE_BASELINE=1`) cannot loosen them.

## API Boundaries

**API Boundary:** the Python `DDMSC` class. Its shapes, dtypes, property names and `DDMSCResult` fields are
documented in `README.md` and typed in `src/fast_deconv/_fast_deconv/ddmsc.pyi`. Changing them breaks DDFacet. Only
`src/python` knows about Python, and it uses only the backend-neutral public API.

**API Boundary:** `algorithm::ddmsc::Ddmsc`, the C++ form of the same contract, used by `replay_ddmsc` and the
non-regression test. It takes host views only; staging into backend memory is its job. On the host backend,
`stage()` borrows instead of copying, so where each input lives depends on who reads it:

| input | cuda | host | must outlive |
|---|---|---|---|
| `raw_psfs`, `xdes`, `mask` | copied to the device, again after `release()` | borrowed | the session, on host |
| `scale_sigmas` | copied to a `std::vector` | copied to a `std::vector` | nothing |
| `scale_bias` | borrowed | borrowed | the session |
| `map_pixel_facet` | borrowed | borrowed | the session |
| `dirty` | copied to the device, residual copied back | written in place | the call |
| `jones_norm`, `weights_freq` | copied to the device per call | borrowed | the call |

On the Python side, `keep_alive` ties every constructor array that may be borrowed to the session.

Everything below `Ddmsc` is internal. Headers under `include/` are public only so that tests and tools can include
them.

**API Boundary:** the `fd_backend` file set. Adding a backend means providing every file below, plus
`binary_dilation`, with the same names and signatures:

| `fd_backend` file | provides |
|---|---|
| `core/exec_impl.hpp` | `exec_resources_impl`, `exec_ctx_impl` |
| `core/memory_types.hpp` | what `span*`/`cont*` mean in this build |
| `core/for_each.hpp`, `core/reduce.hpp` | `for_each_index` (1D to 3D) and `reduce` |
| `linalg/gaussian_convolution.hpp`, `linalg/fft.hpp` | `gaussian_convolution_ctx` and its FFT types |
| `matrix/tiled_argmax_impl.hpp` | the tile cache, checked by `tiled_argmax_backend` |
| `matrix/stats_types.hpp` | the stats reduction types |

## Tooling

- **Build:** CMake with Conan dependencies (emu from the `cosmic` remote); DUCC and Tracy from `FetchContent`.
  `scripts/build.sh` keeps one tree per backend, `build/<build_type>-backend_<backend>`. Wheels are built with
  scikit-build-core, which reads `FAST_DECONV_BACKEND` from the environment. The library is static and linked into
  the Python extension.
- **Verify:** `scripts/verify_all.sh` runs format, clang-tidy, build, tests, compute-sanitizer and ASan/UBSan.
- **Debug on real data:** set `FAST_DECONV_DUMP` to capture a session's inputs to `.npy`, then replay it with
  `replay_ddmsc` (built with `FAST_DECONV_BUILD_TOOLS=ON`) under a debugger, nsys, ncu or Tracy.
