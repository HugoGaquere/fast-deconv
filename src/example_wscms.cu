/// Example: launch run_wscms_cycles with real data dumped from DDFacet.
///
/// Usage:  example_wscms <dump_dir> [--cycle=N] [--device=N]
///
/// `--cycle` selects which `cycle_<N>/` subdirectory to load (default: 1).
/// `--device` selects the CUDA device (default: 0). The example calls
/// cudaSetDevice on that device before any cudaMalloc/cudaMemcpy so the host
/// uploads target the right GPU; wscms::context propagates the same id to
/// core::resources, which binds its stream pool to that device.
///
/// The dump directory must contain `init/` and `cycle_<N>/` subdirectories
/// produced by FastDDFacet's dump_ref utility (set DUMP_REF=<dir> when
/// running DDF.py). `init/` holds the one-time setup (raw PSFs, scale
/// kernels, Wscms ctor inputs, auto-masking thresholds). `cycle_<N>/` holds
/// the per-cycle inputs and runtime parameters (dirty, jones_norm,
/// weights, mask, stop limits, etc.).

#include <cuda_runtime.h>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fast_deconv/algorithm/scales.hpp>
#include <fast_deconv/algorithm/wscms.hpp>
#include <fast_deconv/algorithm/wscms_types.hpp>
#include <fast_deconv/core/resources.hpp>
#include <fast_deconv/core/span_types.hpp>
#include <optional>
#include <string>
#include <vector>

#include "../tests/cpp/npy_loader.hpp"

namespace core = fast_deconv::core;
namespace scale = fast_deconv::scale;
namespace wscms = fast_deconv::algorithm::wscms;

/// Helper: allocate device memory and copy host data into it.
template <typename T>
T* device_upload(const T* host_ptr, std::size_t count)
{
  T* ptr = nullptr;
  cudaMalloc(reinterpret_cast<void**>(&ptr), count * sizeof(T));
  cudaMemcpy(ptr, host_ptr, count * sizeof(T), cudaMemcpyHostToDevice);
  return ptr;
}

/// Treat a NaN scalar from the dump as "field unset" (nullopt).
static std::optional<float> opt_finite(float v)
{
  return std::isnan(v) ? std::nullopt : std::optional<float>(v);
}

int main(int argc, char** argv)
{
  auto usage = [&]() { fprintf(stderr, "Usage: %s <dump_dir> [--cycle=N] [--device=N]\n", argv[0]); };

  std::string dir;
  int cycle_id = 1;
  int device_id = 0;
  for (int i = 1; i < argc; ++i) {
    std::string a(argv[i]);
    if (a.rfind("--cycle=", 0) == 0) {
      cycle_id = std::atoi(a.c_str() + 8);
    } else if (a.rfind("--device=", 0) == 0) {
      device_id = std::atoi(a.c_str() + 9);
    } else if (!a.empty() && a[0] != '-' && dir.empty()) {
      dir = a;
    } else {
      fprintf(stderr, "Unrecognized argument: %s\n", argv[i]);
      usage();
      return 1;
    }
  }
  if (dir.empty()) {
    usage();
    return 1;
  }

  // Bind this thread to the requested device so the raw cudaMalloc/cudaMemcpy
  // calls below target it. wscms::context will pass the same id down to
  // core::resources, which binds its stream pool to the same device.
  cudaError_t set_err = cudaSetDevice(device_id);
  if (set_err != cudaSuccess) {
    fprintf(stderr, "cudaSetDevice(%d) failed: %s\n", device_id, cudaGetErrorString(set_err));
    return 1;
  }

  const std::string cycle_subdir = "/cycle_" + std::to_string(cycle_id) + "/";
  auto load_init = [&](const char* name) { return npy::load_npy(dir + "/init/" + name + ".npy"); };
  auto load_cycle = [&](const char* name) { return npy::load_npy(dir + cycle_subdir + name + ".npy"); };

  // ----- init/ : one-time setup -----
  auto npy_raw_psfs = load_init("raw_psfs");
  auto npy_xdes = load_init("xdes");
  auto npy_scale_sigmas = load_init("scale_sigmas");
  auto npy_scale_bias = load_init("scale_bias");
  auto npy_map_pixel = load_init("map_pixel_facet");
  auto npy_fft_padding = load_init("fft_padding");
  auto npy_gamma = load_init("gamma");
  auto npy_auto_mask_peak_th = load_init("auto_mask_peak_threshold");
  auto npy_auto_mask_rms_th = load_init("auto_mask_rms_threshold");

  // ----- cycle_1/ : per-cycle inputs and params -----
  auto npy_dirty = load_cycle("dirty");
  auto npy_jones_norm = load_cycle("jones_norm");
  auto npy_weights_freq = load_cycle("weights_freq");
  auto npy_mask = load_cycle("mask");

  auto npy_max_iteration = load_cycle("max_iteration");
  auto npy_max_sub_iter = load_cycle("max_sub_iteration");
  auto npy_divergence = load_cycle("divergence_factor");
  auto npy_flux_threshold = load_cycle("flux_threshold");
  auto npy_rms_factor = load_cycle("stop_rms_factor");
  auto npy_stop_peak_factor = load_cycle("stop_peak_factor");
  auto npy_cycle_factor = load_cycle("stop_cycle_factor");
  auto npy_sidelobe_level = load_cycle("stop_sidelobe_level");
  auto npy_stall = load_cycle("stall_threshold");
  auto npy_peak_factor = load_cycle("peak_factor");
  auto npy_clean_negative = load_cycle("clean_negative");
  auto npy_force_auto_mask = load_cycle("force_auto_mask");

  // ----- Extract dimensions from loaded shapes -----
  // dump_ref squeezes arrays, removing the stokes=1 dimension.
  // raw_psfs: (n_facet, n_freq, psf_nrow, psf_ncol)
  const int n_facet = static_cast<int>(npy_raw_psfs.shape[0]);
  const int n_freq = static_cast<int>(npy_raw_psfs.shape[1]);
  const int psf_nrow = static_cast<int>(npy_raw_psfs.shape[2]);
  const int psf_ncol = static_cast<int>(npy_raw_psfs.shape[3]);

  // dirty: (n_freq, nrow, ncol)
  const int nrow = static_cast<int>(npy_dirty.shape[1]);
  const int ncol = static_cast<int>(npy_dirty.shape[2]);

  const int mask_nrow = static_cast<int>(npy_mask.shape[0]);
  const int mask_ncol = static_cast<int>(npy_mask.shape[1]);

  const int n_scales = static_cast<int>(npy_scale_sigmas.shape[0]);
  const int n_order = static_cast<int>(npy_xdes.shape[1]);

  printf("Loaded dump from %s (cycle=%d, device=%d)\n", dir.c_str(), cycle_id, device_id);
  printf("  image: %dx%d  psf: %dx%d  mask: %dx%d  freq: %d  order: %d  facets: %d  scales: %d\n", nrow, ncol, psf_nrow,
         psf_ncol, mask_nrow, mask_ncol, n_freq, n_order, n_facet, n_scales);

  // ----- Upload arrays to device -----
  float* d_dirty = device_upload(npy_dirty.as_float32(), npy_dirty.size());
  float* d_raw_psfs = device_upload(npy_raw_psfs.as_float32(), npy_raw_psfs.size());
  float* d_jones_norm = device_upload(npy_jones_norm.as_float32(), npy_jones_norm.size());
  float* d_xdes = device_upload(npy_xdes.as_float32(), npy_xdes.size());
  float* d_weights = device_upload(npy_weights_freq.as_float32(), npy_weights_freq.size());
  bool* d_mask = device_upload(npy_mask.as_bool(), npy_mask.size());
  float* d_scale_sig = device_upload(npy_scale_sigmas.as_float32(), npy_scale_sigmas.size());

  // Host arrays (no device upload)
  float* h_scale_bias = npy_scale_bias.as_float32();
  int* h_map_pixel = npy_map_pixel.as_int32();

  // ----- Build mdspan views -----
  core::device_span3d<float> dirty(d_dirty, n_freq, nrow, ncol);
  core::device_span4d<float> raw_psfs(d_raw_psfs, n_facet, n_freq, psf_nrow, psf_ncol);
  core::device_span3d<float> jones_norm(d_jones_norm, n_freq, nrow, ncol);
  core::device_span2d<float> xdes(d_xdes, n_freq, n_order);
  core::device_vect<float> weights_freq(d_weights, n_freq);
  core::device_span2d<bool> mask(d_mask, mask_nrow, mask_ncol);
  core::device_vect<float> scale_sigmas(d_scale_sig, n_scales);
  core::host_vect<float> scale_bias(h_scale_bias, n_scales);
  core::host_span2d<int> map_pixel_facet(h_map_pixel, nrow, ncol);

  const float fft_padding = npy_fft_padding.scalar<float>();

  // ----- Build WSCMS context (resources + workspace + FFT plans) -----
  // The mask passed here is the selected cycle's mask, mirroring what Python's
  // set_scale_mask hot-swap would feed in just before this cycle.
  wscms::context ctx(device_id, raw_psfs, xdes, mask, scale_sigmas, scale_bias, map_pixel_facet, nrow, ncol, n_freq,
                     fft_padding);

  // ----- Build WSCMS params from scalars -----
  // The auto-masking thresholds live in init/ (they are deconvolution-wide
  // settings). Master switch is enabled because the dump always contains the
  // threshold files; NaN entries are folded back to nullopt so they do not
  // accidentally short-circuit the per-iter check in run_wscms_cycles.
  wscms::params params{
      .max_iteration = npy_max_iteration.scalar<int>(),
      .divergence_factor = npy_divergence.scalar<float>(),
      .flux_threshold = npy_flux_threshold.scalar<float>(),
      .stop_rms_factor = npy_rms_factor.scalar<float>(),
      .stop_peak_factor = npy_stop_peak_factor.scalar<float>(),
      .stop_cycle_factor = npy_cycle_factor.scalar<float>(),
      .stop_sidelobe_level = npy_sidelobe_level.scalar<float>(),
      .clean_negative = npy_clean_negative.scalar<bool>(),
      .peak_factor = npy_peak_factor.scalar<float>(),
      .gamma = npy_gamma.scalar<float>(),
      .max_clean_iteration = npy_max_sub_iter.scalar<int>(),
      .scale_stall_threshold = npy_stall.scalar<float>(),
      .enable_auto_mask = true,
      .force_enable_auto_mask = npy_force_auto_mask.scalar<bool>(),
      .auto_mask_peak_threshold = opt_finite(npy_auto_mask_peak_th.scalar<float>()),
      .auto_mask_rms_threshold = opt_finite(npy_auto_mask_rms_th.scalar<float>()),
  };

  // ----- Run WSCMS -----
  printf("Running WSCMS on %dx%d image, %d scales, %d freq, %d facets...\n", nrow, ncol, n_scales, n_freq, n_facet);

  const auto t_start = std::chrono::steady_clock::now();
  wscms::wscms_result result = wscms::run_wscms_cycles(ctx, params, dirty, jones_norm, weights_freq);
  cudaDeviceSynchronize();
  const auto t_end = std::chrono::steady_clock::now();
  const double elapsed_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();

  printf("Done in %.3f ms (%.3f s). %zu components (scales=%zu gains=%zu coeffs=%zu)\n", elapsed_ms, elapsed_ms / 1000.0,
         result.peak_coords.size(), result.scales.size(), result.gains.size(), result.coeffs.size());

  const std::size_t n_print = std::min<std::size_t>(5, result.peak_coords.size());
  for (std::size_t i = 0; i < n_print; ++i) {
    printf("  [%zu] (row=%d, col=%d) scale=%d gain=%.6f coeffs=[", i, result.peak_coords[i].first,
           result.peak_coords[i].second, result.scales[i], result.gains[i]);
    for (std::size_t k = 0; k < result.coeffs[i].size(); ++k) {
      printf("%s%.6f", k == 0 ? "" : ", ", result.coeffs[i][k]);
    }
    printf("]\n");
  }

  // ----- Cleanup -----
  cudaFree(d_dirty);
  cudaFree(d_raw_psfs);
  cudaFree(d_jones_norm);
  cudaFree(d_xdes);
  cudaFree(d_weights);
  cudaFree(d_mask);
  cudaFree(d_scale_sig);

  return 0;
}
