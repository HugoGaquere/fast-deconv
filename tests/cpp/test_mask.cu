#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <cstdint>
#include <fast_deconv/common/mask.hpp>
#include <fast_deconv/core/resources.hpp>
#include <fast_deconv/core/span_types.hpp>
#include <fast_deconv/util/cuda_macros.hpp>
#include <utility>
#include <vector>

namespace core = fast_deconv::core;
namespace common = fast_deconv::common;

// With a delta center_psf and sigma=0 for every scale, conv2_psf is a delta,
// the FWHM mask is a single pixel and the dilation is the identity — so the
// final "is-near-component" map matches the per-scale peak premask exactly.
// After finalize_mask_kernel: mask = (!is_near_component) || external_mask.
TEST(BuildIndependantScaleMask, DeltaPsfZeroSigmaProducesPeakOnlyPremask)
{
  const int n_scales = 3;
  const int nrow = 4;
  const int ncol = 5;
  const int plane = nrow * ncol;
  const int total = n_scales * plane;

  // Even psf size keeps the FFT center at (psf_h/2, psf_w/2).
  const int psf_h = 8;
  const int psf_w = 8;
  const int psf_npix = psf_h * psf_w;

  const std::vector<std::pair<int, int>> coords = {{0, 0}, {1, 2}, {3, 4}};
  const std::vector<int> scales = {0, 2, 1};

  core::resources resources(0);
  const auto& sr = resources.get_stream_resources();

  // Output mask buffer, pre-filled with 0xff to verify the internal memset.
  bool* d_mask = resources.alloc_async<bool>(total, sr);
  CHECK_CUDA(cudaMemsetAsync(d_mask, 0xff, total * sizeof(bool), sr.cuda_stream));

  // Delta PSF at FFT-center, single channel.
  const int n_freq = 1;
  std::vector<float> h_psf(n_freq * psf_npix, 0.0f);
  h_psf[(psf_h / 2) * psf_w + (psf_w / 2)] = 1.0f;
  float* d_psf = resources.alloc_async<float>(n_freq * psf_npix, sr);
  CHECK_CUDA(
      cudaMemcpyAsync(d_psf, h_psf.data(), n_freq * psf_npix * sizeof(float), cudaMemcpyHostToDevice, sr.cuda_stream));

  // Single channel: weight = 1.0
  std::vector<float> h_weights(n_freq, 1.0f);
  float* d_weights = resources.alloc_async<float>(n_freq, sr);
  CHECK_CUDA(cudaMemcpyAsync(d_weights, h_weights.data(), n_freq * sizeof(float), cudaMemcpyHostToDevice,
                             sr.cuda_stream));

  // sigma = 0 → freq-domain Gaussian is identically 1, conv2_psf == central_facet_psfs[0].
  float* d_sigmas = resources.alloc_async<float>(n_scales, sr);
  CHECK_CUDA(cudaMemsetAsync(d_sigmas, 0, n_scales * sizeof(float), sr.cuda_stream));

  // External mask all-false (no extra masking) — so output mask = !is_near_component.
  bool* d_external = resources.alloc_async<bool>(plane, sr);
  CHECK_CUDA(cudaMemsetAsync(d_external, 0, plane * sizeof(bool), sr.cuda_stream));

  core::device_span3d<bool> mask_view(d_mask, n_scales, nrow, ncol);
  core::device_span3d<float> psf_view(d_psf, n_freq, psf_h, psf_w);
  core::device_vect<float> weights_view(d_weights, n_freq);
  core::device_vect<float> sigma_view(d_sigmas, n_scales);
  core::device_span2d<bool> external_view(d_external, nrow, ncol);

  const float fft_padding = 1.5f;

  common::build_independant_scale_mask(resources, sr, coords, scales, psf_view, weights_view, sigma_view, fft_padding,
                                       external_view, mask_view);

  std::vector<uint8_t> h_bytes(total);
  CHECK_CUDA(cudaMemcpyAsync(h_bytes.data(), d_mask, total * sizeof(bool), cudaMemcpyDeviceToHost, sr.cuda_stream));
  sr.sync();

  // After finalize: mask = !is_near_component (since external is all-false).
  // Peak coords are the only "near component" pixels → mask=0 there, mask=1 elsewhere.
  std::vector<uint8_t> expected(total, 1);
  for (size_t k = 0; k < coords.size(); ++k) {
    const int s = scales[k];
    const int r = coords[k].first;
    const int c = coords[k].second;
    expected[s * plane + r * ncol + c] = 0;
  }

  for (int i = 0; i < total; ++i) {
    EXPECT_EQ(h_bytes[i], expected[i]) << "Mismatch at flat idx " << i;
  }

  resources.free_async(d_mask, sr);
  resources.free_async(d_psf, sr);
  resources.free_async(d_weights, sr);
  resources.free_async(d_sigmas, sr);
  resources.free_async(d_external, sr);
  sr.sync();
}
