#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <fast_deconv/common/mask.hpp>
#include <fast_deconv/core/memory_types.hpp>
#include <fast_deconv/util/cuda_macros.hpp>
#include <limits>
#include <utility>
#include <vector>

#include "helpers/device_buffers.hpp"
#include "helpers/gpu_test.hpp"
#include "helpers/host_oracles.hpp"

namespace core = fast_deconv::core;
namespace common = fast_deconv::common;
namespace fdtest = fast_deconv::test;

using fdtest::flat;

// ============================================================================
// mask_and_abs_async / mask_less_than_threshold
// Convention everywhere: mask true = excluded/filled, false = valid.
// ============================================================================

class MaskAndAbs : public fdtest::GpuTest {};

TEST_F(MaskAndAbs, Fills2dMaskedPixelsAndTakesAbsOfUnmasked)
{
  const int nrow = 6, ncol = 7, n = nrow * ncol;
  std::vector<float> data(n);
  for (int i = 0; i < n; ++i) data.at(i) = (i % 2 == 0 ? -1.0f : 1.0f) * static_cast<float>(i + 1);
  std::vector<bool> mask(n, false);
  for (int i = 0; i < n; i += 3) mask.at(i) = true;

  const float fill = -std::numeric_limits<float>::infinity();
  const auto sr = res().make_ctx();

  fdtest::device_buffer<float> d_data(res(), sr, data);
  fdtest::device_buffer<bool> d_mask(res(), sr, mask);
  core::device_span2d<float> data_view(d_data.get(), nrow, ncol);
  core::device_span2d<bool> mask_view(d_mask.get(), nrow, ncol);

  common::mask_and_abs_async(sr, data_view, mask_view, fill, /*abs=*/true);
  sr.wait();

  const auto got = d_data.to_host();
  for (int i = 0; i < n; ++i) {
    if (mask.at(i))
      ASSERT_EQ(got.at(i), fill) << "masked pixel " << i;
    else
      ASSERT_FLOAT_EQ(got.at(i), std::fabs(data.at(i))) << "unmasked pixel " << i;
  }
}

TEST_F(MaskAndAbs, WithoutAbsUnmaskedPixelsAreUntouched)
{
  const int nrow = 4, ncol = 5, n = nrow * ncol;
  std::vector<float> data(n);
  for (int i = 0; i < n; ++i) data.at(i) = -0.5f * static_cast<float>(i);
  std::vector<bool> mask(n, false);
  mask.at(3) = true;
  mask.at(12) = true;

  const auto sr = res().make_ctx();
  fdtest::device_buffer<float> d_data(res(), sr, data);
  fdtest::device_buffer<bool> d_mask(res(), sr, mask);
  core::device_span2d<float> data_view(d_data.get(), nrow, ncol);
  core::device_span2d<bool> mask_view(d_mask.get(), nrow, ncol);

  common::mask_and_abs_async(sr, data_view, mask_view, /*fill_value=*/42.0f, /*abs=*/false);
  sr.wait();

  const auto got = d_data.to_host();
  for (int i = 0; i < n; ++i) {
    if (mask.at(i))
      ASSERT_EQ(got.at(i), 42.0f) << "masked pixel " << i;
    else
      ASSERT_EQ(got.at(i), data.at(i)) << "unmasked pixel " << i;  // bit-identical
  }
}

TEST_F(MaskAndAbs, Broadcasts2dMaskAcrossEvery3dSlice)
{
  const int n_batch = 3, nrow = 4, ncol = 5, plane = nrow * ncol;
  std::vector<float> data(n_batch * plane);
  for (std::size_t i = 0; i < data.size(); ++i) data.at(i) = -static_cast<float>(i + 1);
  std::vector<bool> mask(plane, false);
  mask.at(flat(1, 2, ncol)) = true;
  mask.at(flat(3, 4, ncol)) = true;

  const auto sr = res().make_ctx();
  fdtest::device_buffer<float> d_data(res(), sr, data);
  fdtest::device_buffer<bool> d_mask(res(), sr, mask);
  core::device_span3d<float> data_view(d_data.get(), n_batch, nrow, ncol);
  core::device_span2d<bool> mask_view(d_mask.get(), nrow, ncol);

  common::mask_and_abs_async(sr, data_view, mask_view, /*fill_value=*/0.0f, /*abs=*/true);
  sr.wait();

  const auto got = d_data.to_host();
  for (int b = 0; b < n_batch; ++b) {
    for (int i = 0; i < plane; ++i) {
      const int idx = b * plane + i;
      if (mask.at(i))
        ASSERT_EQ(got.at(idx), 0.0f) << "batch " << b << " pixel " << i;
      else
        ASSERT_FLOAT_EQ(got.at(idx), std::fabs(data.at(idx))) << "batch " << b << " pixel " << i;
    }
  }
}

TEST_F(MaskAndAbs, Applies3dMaskPerSliceIndependently)
{
  const int n_batch = 2, nrow = 4, ncol = 5, plane = nrow * ncol;
  std::vector<float> data(n_batch * plane, -1.0f);
  std::vector<bool> mask(n_batch * plane, false);
  // Slice 0: nothing masked. Slice 1: a diagonal-ish pattern.
  for (int i = 0; i < plane; i += 4) mask.at(plane + i) = true;

  const auto sr = res().make_ctx();
  fdtest::device_buffer<float> d_data(res(), sr, data);
  fdtest::device_buffer<bool> d_mask(res(), sr, mask);
  core::device_span3d<float> data_view(d_data.get(), n_batch, nrow, ncol);
  core::device_span3d<bool> mask_view(d_mask.get(), n_batch, nrow, ncol);

  common::mask_and_abs_async(sr, data_view, mask_view, /*fill_value=*/7.0f, /*abs=*/false);
  sr.wait();

  const auto got = d_data.to_host();
  for (int i = 0; i < plane; ++i) ASSERT_EQ(got.at(i), -1.0f) << "slice 0 pixel " << i;  // untouched
  for (int i = 0; i < plane; ++i)
    ASSERT_EQ(got.at(plane + i), mask.at(plane + i) ? 7.0f : -1.0f) << "slice 1 pixel " << i;
}

// The comparison is strict `<`: a pixel exactly equal to the threshold survives.
// This is the sub-clean threshold semantics used by the inner CLEAN loop.
TEST_F(MaskAndAbs, MaskLessThanThresholdIsStrict)
{
  const int nrow = 3, ncol = 5, n = nrow * ncol;
  const float threshold = 0.5f;
  std::vector<float> data = {0.1f,  0.5f, 0.9f, -0.2f, 0.49999f, 0.51f, 0.0f, 2.0f,
                             -5.0f, 0.5f, 0.3f, 0.7f,  0.5f,     1.0f,  -0.5f};
  ASSERT_EQ(data.size(), static_cast<std::size_t>(n));

  const float fill = -1000.0f;
  const auto sr = res().make_ctx();
  fdtest::device_buffer<float> d_data(res(), sr, data);
  core::device_span2d<float> data_view(d_data.get(), nrow, ncol);

  common::mask_less_than_threshold(sr, data_view, threshold, fill);
  sr.wait();

  const auto got = d_data.to_host();
  for (int i = 0; i < n; ++i) {
    if (data.at(i) < threshold)
      ASSERT_EQ(got.at(i), fill) << "pixel " << i;
    else
      ASSERT_EQ(got.at(i), data.at(i)) << "pixel " << i;  // == threshold survives
  }
}

// ============================================================================
// build_auto_mask
// ============================================================================

class BuildAutoMask : public fdtest::GpuTest {};

// With a delta center_psf and sigma=0 for every scale, conv2_psf is a delta,
// the FWHM mask is a single pixel and the dilation is the identity — so the
// final "is-near-component" map matches the per-scale peak premask exactly.
// After finalize_mask_kernel: mask = (!is_near_component) || external_mask.
TEST_F(BuildAutoMask, DeltaPsfZeroSigmaProducesPeakOnlyPremask)
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

  const auto sr = res().make_ctx();

  // Output mask buffer, pre-filled with 0xff to verify the internal memset.
  fdtest::device_buffer<bool> d_mask(res(), sr, static_cast<std::size_t>(total));
  CHECK_CUDA(cudaMemsetAsync(d_mask.get(), 0xff, total * sizeof(bool), sr.cuda_stream));

  // Delta PSF at FFT-center, single channel.
  const int n_freq = 1;
  std::vector<float> h_psf(n_freq * psf_npix, 0.0f);
  h_psf.at((psf_h / 2) * psf_w + (psf_w / 2)) = 1.0f;
  fdtest::device_buffer<float> d_psf(res(), sr, h_psf);

  // Single channel: weight = 1.0
  fdtest::device_buffer<float> d_weights(res(), sr, std::vector<float>(n_freq, 1.0f));

  // sigma = 0 → freq-domain Gaussian is identically 1, conv2_psf == central_facet_psfs[0].
  fdtest::device_buffer<float> d_sigmas(res(), sr, std::vector<float>(n_scales, 0.0f));

  // External mask all-false (no extra masking) — so output mask = !is_near_component.
  fdtest::device_buffer<bool> d_external(res(), sr, std::vector<bool>(plane, false));

  core::device_span3d<bool> mask_view(d_mask.get(), n_scales, nrow, ncol);
  core::device_span3d<float> psf_view(d_psf.get(), n_freq, psf_h, psf_w);
  core::span1d<float> weights_view(d_weights.get(), n_freq);
  core::span1d<float> sigma_view(d_sigmas.get(), n_scales);
  core::device_span2d<bool> external_view(d_external.get(), nrow, ncol);

  const float fft_padding = 1.5f;

  common::build_auto_mask(sr, coords, scales, psf_view, weights_view, sigma_view, fft_padding, external_view,
                          mask_view);
  sr.wait();

  const auto h_bytes = d_mask.to_host();

  // After finalize: mask = !is_near_component (since external is all-false).
  // Peak coords are the only "near component" pixels → mask=0 there, mask=1 elsewhere.
  std::vector<uint8_t> expected(total, 1);
  for (size_t k = 0; k < coords.size(); ++k) {
    const int s = scales.at(k);
    const int r = coords.at(k).first;
    const int c = coords.at(k).second;
    expected.at(s * plane + r * ncol + c) = 0;
  }

  for (int i = 0; i < total; ++i) {
    EXPECT_EQ(h_bytes.at(i), expected.at(i)) << "Mismatch at flat idx " << i;
  }
}

// A real (Gaussian) PSF and a nonzero scale sigma: the component's FWHM
// neighborhood must come out unmasked (false) on that component's scale slice,
// pixels far away stay masked (true), and scales with no components are fully
// masked.
TEST_F(BuildAutoMask, GaussianPsfUnmasksFwhmNeighborhoodOfComponent)
{
  const int n_scales = 2;
  const int nrow = 16, ncol = 16;
  const int plane = nrow * ncol;
  const int psf_h = 16, psf_w = 16;
  const double sigma_psf = 1.5;

  // Single component at the image center, on scale 1 (sigma = 1.0).
  const std::vector<std::pair<int, int>> coords = {{8, 8}};
  const std::vector<int> scales = {1};

  const auto sr = res().make_ctx();

  fdtest::device_buffer<bool> d_mask(res(), sr, static_cast<std::size_t>(n_scales) * plane);
  CHECK_CUDA(cudaMemsetAsync(d_mask.get(), 0xff, n_scales * plane * sizeof(bool), sr.cuda_stream));

  // Gaussian PSF centered at the FFT center (psf_h/2, psf_w/2), single channel.
  const int n_freq = 1;
  const auto h_psf = fdtest::gaussian2d(psf_h, psf_w, psf_h / 2, psf_w / 2, sigma_psf);
  fdtest::device_buffer<float> d_psf(res(), sr, h_psf);
  fdtest::device_buffer<float> d_weights(res(), sr, std::vector<float>(n_freq, 1.0f));
  fdtest::device_buffer<float> d_sigmas(res(), sr, std::vector<float>{0.0f, 1.0f});
  fdtest::device_buffer<bool> d_external(res(), sr, std::vector<bool>(plane, false));

  core::device_span3d<bool> mask_view(d_mask.get(), n_scales, nrow, ncol);
  core::device_span3d<float> psf_view(d_psf.get(), n_freq, psf_h, psf_w);
  core::span1d<float> weights_view(d_weights.get(), n_freq);
  core::span1d<float> sigma_view(d_sigmas.get(), n_scales);
  core::device_span2d<bool> external_view(d_external.get(), nrow, ncol);

  common::build_auto_mask(sr, coords, scales, psf_view, weights_view, sigma_view, 1.5f, external_view, mask_view);
  sr.wait();

  const auto h_mask = d_mask.to_host();

  // Scale 1: the conv2 PSF has sigma_eff = sqrt(sigma_psf^2 + 2*sigma_s^2) ≈ 2.06,
  // FWHM radius ≈ 1.18*sigma_eff ≈ 2.4 px — the component and its immediate
  // neighbors must be valid (false); far corners stay masked (true).
  EXPECT_EQ(h_mask.at(plane + flat(8, 8, ncol)), 0) << "component pixel must be unmasked";
  EXPECT_EQ(h_mask.at(plane + flat(7, 8, ncol)), 0);
  EXPECT_EQ(h_mask.at(plane + flat(8, 9, ncol)), 0);
  EXPECT_EQ(h_mask.at(plane + flat(0, 0, ncol)), 1) << "far pixel must stay masked";
  EXPECT_EQ(h_mask.at(plane + flat(15, 15, ncol)), 1);

  // Scale 0 has no components: its slice is fully masked.
  for (int i = 0; i < plane; ++i) ASSERT_EQ(h_mask.at(i), 1) << "scale-0 pixel " << i;
}

// An externally-masked pixel stays masked even when it hosts a component: the
// external mask is OR'd in after the component-neighborhood negation.
TEST_F(BuildAutoMask, ExternalMaskOverridesComponentNeighborhood)
{
  const int n_scales = 1;
  const int nrow = 4, ncol = 5;
  const int plane = nrow * ncol;
  const int psf_h = 8, psf_w = 8;
  const int psf_npix = psf_h * psf_w;

  const std::vector<std::pair<int, int>> coords = {{1, 2}};
  const std::vector<int> scales = {0};

  const auto sr = res().make_ctx();

  fdtest::device_buffer<bool> d_mask(res(), sr, static_cast<std::size_t>(n_scales) * plane);
  CHECK_CUDA(cudaMemsetAsync(d_mask.get(), 0, n_scales * plane * sizeof(bool), sr.cuda_stream));

  const int n_freq = 1;
  std::vector<float> h_psf(n_freq * psf_npix, 0.0f);
  h_psf.at((psf_h / 2) * psf_w + (psf_w / 2)) = 1.0f;  // delta PSF
  fdtest::device_buffer<float> d_psf(res(), sr, h_psf);
  fdtest::device_buffer<float> d_weights(res(), sr, std::vector<float>(n_freq, 1.0f));
  fdtest::device_buffer<float> d_sigmas(res(), sr, std::vector<float>(n_scales, 0.0f));

  // External mask true exactly at the component pixel.
  std::vector<bool> external(plane, false);
  external.at(flat(1, 2, ncol)) = true;
  fdtest::device_buffer<bool> d_external(res(), sr, external);

  core::device_span3d<bool> mask_view(d_mask.get(), n_scales, nrow, ncol);
  core::device_span3d<float> psf_view(d_psf.get(), n_freq, psf_h, psf_w);
  core::span1d<float> weights_view(d_weights.get(), n_freq);
  core::span1d<float> sigma_view(d_sigmas.get(), n_scales);
  core::device_span2d<bool> external_view(d_external.get(), nrow, ncol);

  common::build_auto_mask(sr, coords, scales, psf_view, weights_view, sigma_view, 1.5f, external_view, mask_view);
  sr.wait();

  const auto h_mask = d_mask.to_host();
  // Delta PSF + sigma 0 would leave exactly the component pixel unmasked, but
  // the external mask overrides it: everything is masked.
  for (int i = 0; i < plane; ++i) ASSERT_EQ(h_mask.at(i), 1) << "pixel " << i;
}
