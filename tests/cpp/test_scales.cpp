#include <gtest/gtest.h>

#include <cmath>
#include <fast_deconv/algorithm/scales.hpp>
#include <fast_deconv/core/memory_types.hpp>
#include <fast_deconv/linalg/fft.hpp>
#include <limits>
#include <random>
#include <utility>
#include <vector>

#include "helpers/backend_test.hpp"
#include "helpers/device_buffers.hpp"
#include "helpers/host_oracles.hpp"
#include "helpers/rng.hpp"

namespace core = fast_deconv::core;
namespace scale = fast_deconv::scale;
namespace linalg = fast_deconv::linalg;
namespace fdtest = fast_deconv::test;

using fdtest::flat;

class ScalesTest : public fdtest::BackendTest {};

// ============================================================================
// make_gaussian_kernels_async — freq-domain Gaussian exp(-2 pi^2 rho^2 sigma^2)
// on the half-complex grid, compared against a host evaluation of the same
// frequency mapping.
// ============================================================================

TEST_F(ScalesTest, GaussianKernelsMatchHostFormulaOnHalfComplexGrid)
{
  const int nrow = 12, ncol_full = 10;
  const int ncol_half = ncol_full / 2 + 1;
  const std::vector<float> sigmas = {0.0f, 1.2f, 2.5f};
  const int n_scales = static_cast<int>(sigmas.size());
  const int slice = nrow * ncol_half;

  const auto sr = res().make_ctx();
  fdtest::device_buffer<float> d_sigmas(sr, sigmas);
  fdtest::device_buffer<float> d_scales(sr, static_cast<std::size_t>(n_scales) * slice);

  core::span1d<float> sigma_view(d_sigmas.get(), n_scales);
  core::span3d<float> scales_view(d_scales.get(), n_scales, nrow, ncol_half);

  scale::make_gaussian_kernels_async(sr, sigma_view, ncol_full, scales_view);
  sr.wait();

  const auto got = d_scales.to_host();
  const double two_pi_sq = 2.0 * 9.869604401089358;  // 2 * pi^2

  for (int s = 0; s < n_scales; ++s) {
    for (int r = 0; r < nrow; ++r) {
      for (int c = 0; c < ncol_half; ++c) {
        const double freq_row =
            r < (nrow + 1) / 2 ? static_cast<double>(r) / nrow : static_cast<double>(r - nrow) / nrow;
        const double freq_col = static_cast<double>(c) / ncol_full;
        const double rhosq = freq_row * freq_row + freq_col * freq_col;
        const double expected = std::exp(-two_pi_sq * rhosq * sigmas.at(s) * sigmas.at(s));
        ASSERT_NEAR(got.at(s * slice + flat(r, c, ncol_half)), expected, 1e-6)
            << "scale " << s << " bin (" << r << ", " << c << ")";
      }
    }
  }

  // sigma = 0 is the identity filter: exactly 1 everywhere.
  for (int i = 0; i < slice; ++i) ASSERT_FLOAT_EQ(got.at(i), 1.0f) << "sigma-0 bin " << i;
}

// ============================================================================
// convolve_with_scales — FFT convolution vs direct host convolution
// ============================================================================

TEST_F(ScalesTest, ConvolveWithScalesMatchesDirectConvolution)
{
  const int nrow = 32, ncol = 32;
  const int npix = nrow * ncol;
  // (n_scales - 1) = 2 must be a multiple of backward_batch_size = 2.
  // Sigma window where the sampled-Gaussian oracle is valid:
  //  - upper bound: the ±5σ support (7.5 px) must fit inside the 8 px pad
  //    margin of padding=1.5, so the circular FFT convolution equals the
  //    zero-padded linear oracle;
  //  - lower bound: the library kernel is the CONTINUOUS Fourier transform
  //    exp(-2π²ρ²σ²) sampled on the grid, which matches a sampled spatial
  //    Gaussian only once it has decayed at Nyquist. For σ=1.0 the Nyquist
  //    value is ~5e-5 and the aliasing shows up as ~2e-4 spatial error;
  //    σ ≥ 1.4 pushes it below 1e-8.
  const std::vector<float> sigmas = {0.0f, 1.4f, 1.5f};
  const int n_scales = static_cast<int>(sigmas.size());

  std::mt19937 rng(103);
  std::vector<float> dirty(npix);
  fdtest::fill_uniform(rng, dirty, -1.0f, 1.0f);

  const auto sr = res().make_ctx();
  linalg::convolve_ctx ctx(sr, nrow, ncol, /*forward_batch=*/1, /*backward_batch=*/n_scales - 1,
                           /*n_backward_plans=*/1, /*padding=*/1.5f);
  fdtest::scoped_work_area wa(sr, ctx);

  fdtest::device_buffer<float> d_sigmas(sr, sigmas);
  fdtest::device_buffer<float> d_kernels(
      sr, static_cast<std::size_t>(n_scales) * ctx.dims().freq_nrow * ctx.dims().freq_ncol);
  core::span1d<float> sigma_view(d_sigmas.get(), n_scales);
  core::span3d<float> kernels_view(d_kernels.get(), n_scales, ctx.dims().freq_nrow, ctx.dims().freq_ncol);
  scale::make_gaussian_kernels_async(sr, sigma_view, ctx.dims().padded_ncol, kernels_view);

  fdtest::device_buffer<float> d_dirty(sr, dirty);
  fdtest::device_buffer<float> d_out(sr, static_cast<std::size_t>(n_scales) * npix);
  core::span2d<float> dirty_view(d_dirty.get(), nrow, ncol);
  core::span3d<float> out_view(d_out.get(), n_scales, nrow, ncol);

  scale::convolve_with_scales(ctx, dirty_view, kernels_view, out_view);
  sr.wait();

  const auto out = d_out.to_host();

  // Scale 0 is a plain device copy of the input.
  for (int i = 0; i < npix; ++i) ASSERT_EQ(out.at(i), dirty.at(i)) << "scale-0 pixel " << i;

  // Scales > 0: FFT round trip vs truncated direct convolution. Tolerance is
  // 1e-4 absolute: fp32 FFT + ±5σ kernel truncation, documented — never expect
  // 1e-6 from this path.
  for (int s = 1; s < n_scales; ++s) {
    const double sigma = sigmas.at(s);
    const int radius = static_cast<int>(std::ceil(5.0 * sigma));
    const int ksize = 2 * radius + 1;
    const auto kernel = fdtest::gaussian2d(ksize, ksize, radius, radius, sigma);
    const auto expected = fdtest::direct_convolve_2d(dirty, nrow, ncol, kernel, ksize, ksize);
    for (int i = 0; i < npix; ++i)
      ASSERT_NEAR(out.at(s * npix + i), expected.at(i), 1e-4f) << "scale " << s << " pixel " << i;
  }
}

// ============================================================================
// scale_selection — biased peak-finding on host_span1d bias
// ============================================================================

class ScaleSelection : public fdtest::BackendTest {
 protected:
  static constexpr int kScales = 3;
  static constexpr int kNrow = 4;
  static constexpr int kNcol = 5;
  static constexpr int kNpix = kNrow * kNcol;

  // Planes of -1 with one planted max per scale.
  std::vector<float> make_planes(const std::vector<float>& maxes)
  {
    std::vector<float> planes(kScales * kNpix, -1.0f);
    for (int s = 0; s < kScales; ++s) planes.at(s * kNpix + flat(s, s, kNcol)) = maxes.at(s);
    return planes;
  }

  int run(std::vector<float> planes, std::vector<float> bias, const std::vector<int>& retired)
  {
    const auto sr = res().make_ctx();
    fdtest::device_buffer<float> d_planes(sr, planes);
    core::span3d<float> planes_view(d_planes.get(), kScales, kNrow, kNcol);
    core::host_span1d<float> bias_view(bias.data(), kScales);
    const int best = scale::scale_selection(sr, planes_view, bias_view, retired);
    sr.wait();
    return best;
  }
};

TEST_F(ScaleSelection, BiasIsMultiplicative)
{
  // Raw maxes {2, 5, 3} x bias {1.0, 0.5, 1.0} → biased {2, 2.5, 3} → scale 2.
  // An additive bias would give {3, 5.5, 4} and pick scale 1 instead.
  EXPECT_EQ(run(make_planes({2.0f, 5.0f, 3.0f}), {1.0f, 0.5f, 1.0f}, {}), 2);
}

TEST_F(ScaleSelection, RetiredScalesAreExcluded)
{
  EXPECT_EQ(run(make_planes({2.0f, 5.0f, 3.0f}), {1.0f, 0.5f, 1.0f}, {2}), 1);
  EXPECT_EQ(run(make_planes({2.0f, 5.0f, 3.0f}), {1.0f, 0.5f, 1.0f}, {1, 2}), 0);
}

// A plane filled with -inf — exactly what mask_and_abs_async leaves behind for
// a fully-masked scale — can never win the selection.
TEST_F(ScaleSelection, FullyMaskedPlaneNeverWins)
{
  auto planes = make_planes({2.0f, 5.0f, 3.0f});
  std::fill(planes.begin() + 1 * kNpix, planes.begin() + 2 * kNpix, -std::numeric_limits<float>::infinity());
  EXPECT_EQ(run(planes, {1.0f, 1.0f, 0.1f}, {}), 0);  // biased: {2, -inf, 0.3}
}

// Pin the current fallback: when every scale is retired the loop never updates
// best_scale and the function returns 0 (scales.cu default init) — arguably a
// bug, but downstream relies on getting a valid index back.
TEST_F(ScaleSelection, AllRetiredReturnsScaleZero)
{
  EXPECT_EQ(run(make_planes({2.0f, 5.0f, 3.0f}), {1.0f, 1.0f, 1.0f}, {0, 1, 2}), 0);
}

// ============================================================================
// convolve_psfs_with_scale(s)_async — delta PSFs make the outputs analytic:
// conv_psf = G(sigma), conv2_mean = sum_f w[f] * G(sigma * sqrt(2)).
// ============================================================================

class ConvolvePsfs : public fdtest::BackendTest {
 protected:
  static constexpr int kFacets = 2;
  static constexpr int kFreq = 2;
  static constexpr int kH = 32;
  static constexpr int kW = 32;
  static constexpr int kNpix = kH * kW;

  std::vector<float> delta_psfs() const
  {
    std::vector<float> psfs(kFacets * kFreq * kNpix, 0.0f);
    for (int b = 0; b < kFacets; ++b)
      for (int f = 0; f < kFreq; ++f) psfs.at((b * kFreq + f) * kNpix + flat(kH / 2, kW / 2, kW)) = 1.0f;
    return psfs;
  }

  // Uploads the scene, runs the single-scale entry point at @p scale_idx (or
  // the all-scales one when @p scale_idx < 0) and returns {conv_psf, conv2_mean}.
  // Both outputs carry a leading scale axis in the all-scales case.
  std::pair<std::vector<float>, std::vector<float>> run(const std::vector<float>& psfs,
                                                        const std::vector<float>& sigmas,
                                                        const std::vector<float>& weights, int scale_idx)
  {
    const int n_scales = scale_idx < 0 ? static_cast<int>(sigmas.size()) : 1;
    const auto sr = res().make_ctx();
    linalg::convolve_ctx ctx(sr, kH, kW, /*forward_batch=*/kFreq, /*backward_batch=*/kFreq,
                             /*n_backward_plans=*/2, /*padding=*/1.5f);
    fdtest::scoped_work_area wa(sr, ctx);

    fdtest::device_buffer<float> d_psfs(sr, psfs);
    fdtest::device_buffer<float> d_sigmas(sr, sigmas);
    fdtest::device_buffer<float> d_w(sr, weights);
    fdtest::device_buffer<float> d_conv(sr, static_cast<std::size_t>(n_scales) * psfs.size());
    fdtest::device_buffer<float> d_conv2(sr, static_cast<std::size_t>(n_scales) * kFacets * kNpix);

    core::span4d<float> psf_view(d_psfs.get(), kFacets, kFreq, kH, kW);
    core::span1d<float> sigma_view(d_sigmas.get(), static_cast<int>(sigmas.size()));
    core::span1d<float> w_view(d_w.get(), kFreq);

    if (scale_idx < 0) {
      core::span5d<float> conv_view(d_conv.get(), n_scales, kFacets, kFreq, kH, kW);
      core::span4d<float> conv2_view(d_conv2.get(), n_scales, kFacets, kH, kW);
      scale::convolve_psfs_with_scales_async(ctx, psf_view, sigma_view, w_view, conv_view, conv2_view);
    } else {
      core::span4d<float> conv_view(d_conv.get(), kFacets, kFreq, kH, kW);
      core::span3d<float> conv2_view(d_conv2.get(), kFacets, kH, kW);
      scale::convolve_psfs_with_scale_async(ctx, psf_view, sigma_view, scale_idx, w_view, conv_view, conv2_view);
    }
    sr.wait();

    return {d_conv.to_host(), d_conv2.to_host()};
  }
};

// Single-scale entry point at both branches: scale 0 is a copy plus the channel
// mean, scale 1 convolves. Delta PSFs make the scale-1 outputs analytic —
// conv_psf = G(sigma), conv2_mean = G(sigma * sqrt(2)) since the weights sum to 1.
TEST_F(ConvolvePsfs, SingleScaleEntryPointCopiesAtZeroAndConvolvesAbove)
{
  const std::vector<float> weights = {0.6f, 0.4f};
  const double sigma = 1.2;

  std::mt19937 rng(107);
  std::vector<float> random_psfs(kFacets * kFreq * kNpix);
  fdtest::fill_uniform(rng, random_psfs, 0.0f, 1.0f);

  {
    const auto [conv, conv2] = run(random_psfs, {0.0f}, weights, /*scale_idx=*/0);
    for (std::size_t i = 0; i < random_psfs.size(); ++i)
      ASSERT_EQ(conv.at(i), random_psfs.at(i)) << "conv_psf flat " << i;
    for (int b = 0; b < kFacets; ++b) {
      const std::vector<float> facet(random_psfs.begin() + b * kFreq * kNpix,
                                     random_psfs.begin() + (b + 1) * kFreq * kNpix);
      const auto expected = fdtest::weighted_sum(facet, weights, kNpix);
      for (int i = 0; i < kNpix; ++i)
        ASSERT_NEAR(conv2.at(b * kNpix + i), expected.at(i), 1e-6f) << "facet " << b << " pixel " << i;
    }
  }

  const auto [conv, conv2] = run(delta_psfs(), {static_cast<float>(sigma)}, weights, /*scale_idx=*/1);
  const auto g1 = fdtest::gaussian2d(kH, kW, kH / 2, kW / 2, sigma);
  const auto g2 = fdtest::gaussian2d(kH, kW, kH / 2, kW / 2, sigma * std::sqrt(2.0));
  for (int b = 0; b < kFacets; ++b) {
    for (int f = 0; f < kFreq; ++f)
      for (int i = 0; i < kNpix; ++i)
        ASSERT_NEAR(conv.at((b * kFreq + f) * kNpix + i), g1.at(i), 1e-4f)
            << "facet " << b << " freq " << f << " pixel " << i;
    for (int i = 0; i < kNpix; ++i)
      ASSERT_NEAR(conv2.at(b * kNpix + i), g2.at(i), 1e-4f) << "facet " << b << " pixel " << i;
  }
}

TEST_F(ConvolvePsfs, AllScalesVariantSlicesPerScaleOutputs)
{
  const auto psfs = delta_psfs();
  const std::vector<float> weights = {0.6f, 0.4f};
  const std::vector<float> sigmas = {0.0f, 1.2f};
  const int n_scales = static_cast<int>(sigmas.size());

  const auto [conv, conv2] = run(psfs, sigmas, weights, /*scale_idx=*/-1);

  const std::size_t conv_scale_stride = psfs.size();
  const std::size_t conv2_scale_stride = static_cast<std::size_t>(kFacets) * kNpix;

  // Scale 0: identity copy of the delta PSFs, channel-mean delta.
  for (std::size_t i = 0; i < conv_scale_stride; ++i) ASSERT_EQ(conv.at(i), psfs.at(i)) << "scale-0 conv flat " << i;
  for (int b = 0; b < kFacets; ++b)
    for (int i = 0; i < kNpix; ++i)
      ASSERT_NEAR(conv2.at(b * kNpix + i), psfs.at(b * kFreq * kNpix + i), 1e-6f) << "scale-0 conv2 " << i;

  // Scale 1: Gaussian / sqrt(2)-Gaussian at the delta position.
  const auto g1 = fdtest::gaussian2d(kH, kW, kH / 2, kW / 2, sigmas.at(1));
  const auto g2 = fdtest::gaussian2d(kH, kW, kH / 2, kW / 2, sigmas.at(1) * std::sqrt(2.0));
  for (int b = 0; b < kFacets; ++b) {
    for (int f = 0; f < kFreq; ++f)
      for (int i = 0; i < kNpix; ++i)
        ASSERT_NEAR(conv.at(conv_scale_stride + (b * kFreq + f) * kNpix + i), g1.at(i), 1e-4f)
            << "scale-1 conv facet " << b << " freq " << f << " pixel " << i;
    for (int i = 0; i < kNpix; ++i)
      ASSERT_NEAR(conv2.at(conv2_scale_stride + b * kNpix + i), g2.at(i), 1e-4f)
          << "scale-1 conv2 facet " << b << " pixel " << i;
  }
}
