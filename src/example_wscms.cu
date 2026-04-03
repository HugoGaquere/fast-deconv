/// Example: launch run_wscms with synthetic data on GPU.

#include <cuda_runtime.h>

#include <cstdio>
#include <cstring>
#include <fast_deconv/algorithm/wscms.hpp>
#include <fast_deconv/algorithm/wscms_types.hpp>
#include <fast_deconv/core/logger.hpp>
#include <fast_deconv/core/span_types.hpp>
#include <vector>

namespace core = fast_deconv::core;
namespace wscms = fast_deconv::algorithm::wscms;

/// Helper: allocate device memory and zero it.
template <typename T>
T* device_alloc_zero(std::size_t count)
{
  T* ptr = nullptr;
  cudaMalloc(reinterpret_cast<void**>(&ptr), count * sizeof(T));
  cudaMemset(ptr, 0, count * sizeof(T));
  return ptr;
}

int main()
{
  // ----- Dimensions -----
  constexpr int n_freq = 2;    // spectral channels
  constexpr int n_facet = 1;   // facets
  constexpr int nrow = 1024;   // image rows
  constexpr int ncol = 1024;   // image cols
  constexpr int n_scales = 3;  // number of scales

  // PSF has shape (n_freq, n_facet, n_freq, n_facet, psf_nrow, psf_ncol)
  // psf spatial size is typically 2*nrow-1, 2*ncol-1 but use nrow/ncol for simplicity
  constexpr int psf_nrow = nrow;
  constexpr int psf_ncol = ncol;

  // ----- Allocate device arrays -----

  // dirty: (n_freq, n_facet, nrow, ncol)
  float* d_dirty = device_alloc_zero<float>(n_freq * n_facet * nrow * ncol);

  // mean_residual: (nrow, ncol)
  float* d_mean_residual = device_alloc_zero<float>(nrow * ncol);

  // psfs: (n_freq, n_facet, n_freq, n_facet, psf_nrow, psf_ncol)
  float* d_psfs =
      device_alloc_zero<float>(n_freq * n_facet * n_freq * n_facet * psf_nrow * psf_ncol);

  // psfs_2: (n_scales, n_facet, psf_nrow, psf_ncol)
  float* d_psfs_2 = device_alloc_zero<float>(n_scales * n_facet * psf_nrow * psf_ncol);

  // jones_norm: (n_freq, n_facet, nrow, ncol) — fill with 1.0
  float* d_jones_norm = device_alloc_zero<float>(n_freq * n_facet * nrow * ncol);
  {
    std::vector<float> ones(n_freq * n_facet * nrow * ncol, 1.0f);
    cudaMemcpy(d_jones_norm, ones.data(), ones.size() * sizeof(float), cudaMemcpyHostToDevice);
  }

  // xdes: (n_freq, n_freq) — spectral design matrix (identity)
  float* d_xdes = device_alloc_zero<float>(n_freq * n_freq);
  {
    std::vector<float> identity(n_freq * n_freq, 0.0f);
    for (int i = 0; i < n_freq; i++) identity[i * n_freq + i] = 1.0f;
    cudaMemcpy(d_xdes, identity.data(), identity.size() * sizeof(float), cudaMemcpyHostToDevice);
  }

  // weights_freq: (n_freq)
  float* d_weights_freq = device_alloc_zero<float>(n_freq);
  {
    std::vector<float> w(n_freq, 1.0f);
    cudaMemcpy(d_weights_freq, w.data(), w.size() * sizeof(float), cudaMemcpyHostToDevice);
  }

  // scale_masks: (n_scales, nrow * ncol) stored as 2D — all true
  bool* d_scale_masks = nullptr;
  cudaMalloc(reinterpret_cast<void**>(&d_scale_masks), n_scales * nrow * ncol * sizeof(bool));
  cudaMemset(d_scale_masks, 1, n_scales * nrow * ncol * sizeof(bool));

  // scale_sigmas: (n_scales) on device
  float* d_scale_sigmas = device_alloc_zero<float>(n_scales);
  {
    std::vector<float> sigmas = {0.0f, 1.0f, 2.0f};
    cudaMemcpy(d_scale_sigmas, sigmas.data(), sigmas.size() * sizeof(float),
               cudaMemcpyHostToDevice);
  }

  // scale_bias: (n_scales) on host
  std::vector<float> h_scale_bias(n_scales, 1.0f);

  // map_pixel_facet: (nrow, ncol) on host — all zeros (single facet)
  std::vector<int> h_map_pixel_facet(nrow * ncol, 0);

  // gains: (n_scales, n_facet) on host
  std::vector<float> h_gains(n_scales * n_facet, 0.5f);

  // ----- Build mdspan views -----
  core::device_span4d<float> dirty(d_dirty, n_freq, n_facet, nrow, ncol);
  core::device_span2d<float> mean_residual(d_mean_residual, nrow, ncol);
  core::device_span6d<float> psfs(d_psfs, n_freq, n_facet, n_freq, n_facet, psf_nrow, psf_ncol);
  core::device_span4d<float> psfs_2(d_psfs_2, n_scales, n_facet, psf_nrow, psf_ncol);

  // Context arrays
  core::device_span4d<float> jones_norm(d_jones_norm, n_freq, n_facet, nrow, ncol);
  core::device_span2d<float> xdes(d_xdes, n_freq, n_freq);
  core::device_vect<float> weights_freq(d_weights_freq, n_freq);
  core::device_span2d<bool> scale_masks(d_scale_masks, n_scales, nrow * ncol);
  core::device_vect<float> scale_sigmas(d_scale_sigmas, n_scales);
  core::host_vect<float> scale_bias(h_scale_bias.data(), n_scales);
  core::host_span2d<int> map_pixel_facet(h_map_pixel_facet.data(), nrow, ncol);
  core::host_span2d<float> gains(h_gains.data(), n_scales, n_facet);

  // ----- Build WSCMS context and params -----
  wscms::WSCMS_ctx ctx{
      .jones_norm = jones_norm,
      .xdes = xdes,
      .weights_freq = weights_freq,
      .scale_masks = scale_masks,
      .scale_sigmas = scale_sigmas,
      .scale_bias = scale_bias,
      .map_pixel_facet = map_pixel_facet,
      .gains = gains,
  };

  wscms::WSCMS_params params{
      .clean_negative = true,
      .peak_factor = 0.1f,
      .max_iteration = 100,
      .n_scales = n_scales,
  };

  const fast_deconv::algorithm::wscms::scale_convole_ctx scale_ctx =
      fast_deconv::algorithm::wscms::detail::make_scale_convolve_ctx(nrow, ncol, n_scales, 1.5f);

  fast_deconv::core::resources resources(0);

  // ----- Run WSCMS -----
  printf("Running WSCMS on %dx%d image, %d scales, %d freq, %d facet...\n", nrow, ncol, n_scales,
         n_freq, n_facet);

  wscms::run_wscms(resources, dirty, mean_residual, psfs, psfs_2, ctx, scale_ctx, params);

  cudaDeviceSynchronize();
  printf("Done.\n");

  // ----- Cleanup -----
  cudaFree(d_dirty);
  cudaFree(d_mean_residual);
  cudaFree(d_psfs);
  cudaFree(d_psfs_2);
  cudaFree(d_jones_norm);
  cudaFree(d_xdes);
  cudaFree(d_weights_freq);
  cudaFree(d_scale_masks);
  cudaFree(d_scale_sigmas);

  return 0;
}
