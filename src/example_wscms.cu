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
  constexpr int n_freq = 2;     // spectral channels
  constexpr int n_stokes = 1;   // Stokes parameters
  constexpr int n_facet = 121;    // facets
  constexpr int nrow = 20000;    // image rows
  constexpr int ncol = 20000;    // image cols
  constexpr int n_scales = 4;   // number of scales

  constexpr int psf_nrow = 1700;
  constexpr int psf_ncol = 1700;

  // ----- Allocate device arrays -----

  // dirty: (n_freq, n_stokes, nrow, ncol)
  float* d_dirty = device_alloc_zero<float>(n_freq * n_stokes * nrow * ncol);

  // raw_psfs: (n_facet, n_freq, n_stokes, psf_nrow, psf_ncol)
  float* d_raw_psfs =
      device_alloc_zero<float>(n_facet * n_freq * n_stokes * psf_nrow * psf_ncol);

  // jones_norm: (n_freq, n_stokes, nrow, ncol) — fill with 1.0
  float* d_jones_norm = device_alloc_zero<float>(n_freq * n_stokes * nrow * ncol);
  {
    std::vector<float> ones(n_freq * n_stokes * nrow * ncol, 1.0f);
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

  // scale_masks: (n_scales, nrow * ncol) — all false (no pixel masked)
  bool* d_scale_masks = nullptr;
  cudaMalloc(reinterpret_cast<void**>(&d_scale_masks), n_scales * nrow * ncol * sizeof(bool));
  cudaMemset(d_scale_masks, 0, n_scales * nrow * ncol * sizeof(bool));

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

  // mask: (nrow * ncol) — no pixel masked
  bool* d_mask = nullptr;
  cudaMalloc(reinterpret_cast<void**>(&d_mask), nrow * ncol * sizeof(bool));
  cudaMemset(d_mask, 0, nrow * ncol * sizeof(bool));

  // ----- Build mdspan views -----
  core::device_span4d<float> dirty(d_dirty, n_freq, n_stokes, nrow, ncol);
  core::device_span5d<float> raw_psfs(d_raw_psfs, n_facet, n_freq, n_stokes, psf_nrow, psf_ncol);

  core::device_span4d<float> jones_norm(d_jones_norm, n_freq, n_stokes, nrow, ncol);
  core::device_span2d<float> xdes(d_xdes, n_freq, n_freq);
  core::device_vect<float> weights_freq(d_weights_freq, n_freq);
  core::device_span2d<bool> scale_masks(d_scale_masks, n_scales, nrow * ncol);
  core::device_vect<float> scale_sigmas(d_scale_sigmas, n_scales);
  core::host_vect<float> scale_bias(h_scale_bias.data(), n_scales);
  core::host_span2d<int> map_pixel_facet(h_map_pixel_facet.data(), nrow, ncol);

  // ----- Build WSCMS context and params -----
  wscms::WSCMS_ctx ctx{
      .raw_psfs = raw_psfs,
      .xdes = xdes,
      .scale_masks = scale_masks,
      .scale_sigmas = scale_sigmas,
      .scale_bias = scale_bias,
      .map_pixel_facet = map_pixel_facet,
  };

  wscms::WSCMS_params params{
      .clean_negative = true,
      .peak_factor = 0.1f,
      .gamma = 0.5f,
      .max_sub_iteration = 100,
      .n_scales = n_scales,
      .stop_flux = 0.0f,
      .max_iteration = 1000,
      .divergence_factor = 1.5f,
      .stall_threshold = 1e-6f,
      .forbidden_scales = {},
  };

  const wscms::scale_convole_ctx scale_ctx =
      wscms::detail::make_scale_convolve_ctx(nrow, ncol, n_scales, 1.5f);
  const wscms::psf_convolve_ctx psf_ctx =
      wscms::detail::make_psf_convolve_ctx(psf_nrow, psf_ncol, n_freq, 1.5f);

  core::resources resources(0);

  // ----- Run WSCMS -----
  printf("Running WSCMS on %dx%d image, %d scales, %d freq, %d facet...\n", nrow, ncol, n_scales,
         n_freq, n_facet);

  wscms::wscms_result result =
      wscms::run_wscms(resources, dirty, jones_norm, weights_freq, ctx, scale_ctx, psf_ctx,
                       d_mask, params);

  cudaDeviceSynchronize();
  printf("Done. %d components, %d iterations.\n",
         static_cast<int>(result.components.size()), result.total_iterations);

  // ----- Cleanup -----
  cudaFree(d_dirty);
  cudaFree(d_raw_psfs);
  cudaFree(d_jones_norm);
  cudaFree(d_xdes);
  cudaFree(d_weights_freq);
  cudaFree(d_scale_masks);
  cudaFree(d_scale_sigmas);
  cudaFree(d_mask);

  return 0;
}
