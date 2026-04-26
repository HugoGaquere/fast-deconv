/// Example: launch run_wscms_cycles with real data dumped from DDFacet.
///
/// Usage:  example_wscms <dump_dir>
///
/// The dump directory must contain .npy files produced by FastDDFacet's
/// dump_ref utility (set DUMP_REF=<dir> when running DDF.py).

#include <cuda_runtime.h>

#include <cstdio>
#include <cstring>
#include <fast_deconv/algorithm/scales.hpp>
#include <fast_deconv/algorithm/wscms.hpp>
#include <fast_deconv/algorithm/wscms_types.hpp>
#include <fast_deconv/core/resources.hpp>
#include <fast_deconv/core/span_types.hpp>
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

int main(int argc, char** argv)
{
  if (argc < 2) {
    fprintf(stderr, "Usage: %s <dump_dir>\n", argv[0]);
    return 1;
  }
  const std::string dir(argv[1]);
  auto load = [&](const char* name) { return npy::load_npy(dir + "/" + name + ".npy"); };

  // ----- Load WSCMS_ctx inputs -----
  auto npy_raw_psfs = load("raw_psfs");
  auto npy_xdes = load("xdes");
  auto npy_scale_masks = load("mask");
  auto npy_scale_sigmas = load("scale_sigmas");
  auto npy_scale_bias = load("scale_bias");
  auto npy_map_pixel = load("map_pixel_facet");

  // ----- Load scalar parameters -----
  auto npy_gamma = load("gamma");
  auto npy_peak_factor = load("peak_factor");
  auto npy_clean_negative = load("clean_negative");
  auto npy_fft_padding = load("fft_padding");

  // ----- Load run inputs -----
  auto npy_dirty = load("dirty");
  auto npy_jones_norm = load("jones_norm");
  auto npy_weights_freq = load("weights_freq");
  auto npy_stop_flux = load("stop_flux");
  auto npy_max_iteration = load("max_iteration");
  auto npy_max_sub_iter = load("max_sub_iteration");
  auto npy_divergence = load("divergence_factor");
  auto npy_stall = load("stall_threshold");

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

  const int mask_nrow = static_cast<int>(npy_scale_masks.shape[0]);
  const int mask_ncol = static_cast<int>(npy_scale_masks.shape[1]);

  const int n_scales = static_cast<int>(npy_scale_sigmas.shape[0]);
  const int n_order = static_cast<int>(npy_xdes.shape[1]);

  printf("Loaded dump from %s\n", dir.c_str());
  printf("  image: %dx%d  psf: %dx%d mask: %dx%d  freq: %d  order: %d  facets: %d  scales: %d\n", nrow, ncol, psf_nrow,
         psf_ncol, mask_nrow, mask_ncol, n_freq, n_order, n_facet, n_scales);

  // ----- Upload arrays to device -----
  float* d_dirty = device_upload(npy_dirty.as_float32(), npy_dirty.size());
  float* d_raw_psfs = device_upload(npy_raw_psfs.as_float32(), npy_raw_psfs.size());
  float* d_jones_norm = device_upload(npy_jones_norm.as_float32(), npy_jones_norm.size());
  float* d_xdes = device_upload(npy_xdes.as_float32(), npy_xdes.size());
  float* d_weights = device_upload(npy_weights_freq.as_float32(), npy_weights_freq.size());
  bool* d_scale_masks = device_upload(npy_scale_masks.as_bool(), npy_scale_masks.size());
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
  core::device_span2d<bool> scale_masks(d_scale_masks, mask_nrow, mask_ncol);
  core::device_vect<float> scale_sigmas(d_scale_sig, n_scales);
  core::host_vect<float> scale_bias(h_scale_bias, n_scales);
  core::host_span2d<int> map_pixel_facet(h_map_pixel, nrow, ncol);

  const float fft_padding = npy_fft_padding.scalar<float>();

  // ----- Build WSCMS context (resources + workspace + FFT plans) -----
  wscms::context ctx(/*exec_device=*/0, raw_psfs, xdes, scale_masks, scale_sigmas, scale_bias,
                     map_pixel_facet, nrow, ncol, n_freq, fft_padding);

  // ----- Build WSCMS params from scalars -----
  wscms::params params{
      .max_iteration = 5,  // npy_max_iteration.scalar<int>(),
      .stop_flux_threshold = npy_stop_flux.scalar<float>(),
      .divergence_factor = npy_divergence.scalar<float>(),
      .scale_stall_threshold = npy_stall.scalar<float>(),
      .clean_negative = npy_clean_negative.scalar<bool>(),
      .peak_factor = npy_peak_factor.scalar<float>(),
      .gamma = npy_gamma.scalar<float>(),
      .max_clean_iteration = npy_max_sub_iter.scalar<int>(),
  };

  // ----- Run WSCMS -----
  printf("Running WSCMS on %dx%d image, %d scales, %d freq, %d facets...\n", nrow, ncol, n_scales, n_freq, n_facet);

  wscms::wscms_result result =
      wscms::run_wscms_cycles(ctx, params, dirty, jones_norm, weights_freq);

  cudaDeviceSynchronize();
  printf("Done. %zu components (scales=%zu gains=%zu coeffs=%zu)\n", result.peak_coords.size(),
         result.scales.size(), result.gains.size(), result.coeffs.size());

  const std::size_t n_print = std::min<std::size_t>(5, result.peak_coords.size());
  for (std::size_t i = 0; i < n_print; ++i) {
    printf("  [%zu] (row=%d, col=%d) scale=%d gain=%.6f coeffs=[", i,
           result.peak_coords[i].first, result.peak_coords[i].second, result.scales[i],
           result.gains[i]);
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
  cudaFree(d_scale_masks);
  cudaFree(d_scale_sig);

  return 0;
}
