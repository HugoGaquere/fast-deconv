#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <fast_deconv/common/clean.hpp>
#include <fast_deconv/core/span_types.hpp>
#include <random>
#include <utility>
#include <vector>

#include "helpers/device_buffers.hpp"
#include "helpers/gpu_test.hpp"
#include "helpers/host_oracles.hpp"
#include "helpers/rng.hpp"

namespace core = fast_deconv::core;
namespace common = fast_deconv::common;
namespace fdtest = fast_deconv::test;

using fdtest::flat;

namespace {

// Host oracle mirroring the kernel's centering convention: the PSF's
// (ph/2, pw/2) pixel lands on the peak; out-of-image parts are clipped.
// residual[y, x] -= coeff * gain * psf[i, j] with (y, x) = peak + (i - ph/2, j - pw/2).
std::vector<float> host_subtract(const std::vector<float>& residual, int h, int w, const std::vector<float>& psf,
                                 int ph, int pw, std::pair<int, int> peak, float gain, float coeff = 1.0f)
{
  std::vector<float> out = residual;
  const int y0 = peak.first - ph / 2;
  const int x0 = peak.second - pw / 2;
  for (int i = 0; i < ph; ++i) {
    for (int j = 0; j < pw; ++j) {
      const int y = y0 + i, x = x0 + j;
      if (y < 0 || y >= h || x < 0 || x >= w) continue;
      out.at(flat(y, x, w)) -= coeff * gain * psf.at(flat(i, j, pw));
    }
  }
  return out;
}

}  // namespace

class SubtractComponent : public fdtest::GpuTest {
 protected:
  static constexpr int kH = 16;
  static constexpr int kW = 20;  // non-square residual

  std::vector<float> run_2d(const std::vector<float>& residual, const std::vector<float>& psf, int ph, int pw,
                            std::pair<int, int> peak, float gain)
  {
    const auto sr = res().make_stream();
    fdtest::device_buffer<float> d_res(res(), sr, residual);
    fdtest::device_buffer<float> d_psf(res(), sr, psf);

    core::device_span2d<float> res_view(d_res.get(), kH, kW);
    core::device_span2d<float> psf_view(d_psf.get(), ph, pw);

    common::subtract_component_async(sr, res_view, psf_view, peak, gain);
    sr.sync();
    return d_res.to_host();
  }

  void expect_matches_oracle(const std::vector<float>& got, const std::vector<float>& expected)
  {
    ASSERT_EQ(got.size(), expected.size());
    for (std::size_t i = 0; i < got.size(); ++i) ASSERT_FLOAT_EQ(got.at(i), expected.at(i)) << "flat index " << i;
  }
};

TEST_F(SubtractComponent, InteriorPeakSubtractsScaledPsfWindow)
{
  std::mt19937 rng(71);
  std::vector<float> residual(kH * kW);
  fdtest::fill_uniform(rng, residual, -1.0f, 1.0f);
  std::vector<float> psf(5 * 5);
  fdtest::fill_uniform(rng, psf, 0.0f, 1.0f);

  const std::pair<int, int> peak{8, 10};
  const float gain = 0.3f;

  const auto got = run_2d(residual, psf, 5, 5, peak, gain);
  const auto expected = host_subtract(residual, kH, kW, psf, 5, 5, peak, gain);
  expect_matches_oracle(got, expected);

  // Pixels outside the 5x5 window are bit-identical to the input.
  for (int y = 0; y < kH; ++y)
    for (int x = 0; x < kW; ++x)
      if (std::abs(y - peak.first) > 2 || std::abs(x - peak.second) > 2)
        ASSERT_EQ(got.at(flat(y, x, kW)), residual.at(flat(y, x, kW))) << "touched (" << y << ", " << x << ")";
}

TEST_F(SubtractComponent, PeaksAtCornersAndEdgesClipWithoutOutOfBounds)
{
  std::mt19937 rng(73);
  std::vector<float> residual(kH * kW);
  fdtest::fill_uniform(rng, residual, -1.0f, 1.0f);
  std::vector<float> psf(5 * 5);
  fdtest::fill_uniform(rng, psf, 0.0f, 1.0f);

  for (const auto& peak : {std::pair<int, int>{0, 0}, {0, kW - 1}, {kH - 1, 0}, {kH - 1, kW - 1}}) {
    const auto got = run_2d(residual, psf, 5, 5, peak, 0.5f);
    const auto expected = host_subtract(residual, kH, kW, psf, 5, 5, peak, 0.5f);
    expect_matches_oracle(got, expected);
  }
}

// Even-sized PSF pins the b/2 centering convention: psf pixel (4, 4) of an 8x8
// PSF lands on the peak, so the window is asymmetric (4 above/left, 3 below/right).
TEST_F(SubtractComponent, EvenSizedPsfPinsCenterConvention)
{
  std::mt19937 rng(79);
  std::vector<float> residual(kH * kW);
  fdtest::fill_uniform(rng, residual, -1.0f, 1.0f);
  std::vector<float> psf(8 * 8);
  fdtest::fill_uniform(rng, psf, 0.0f, 1.0f);

  const std::pair<int, int> peak{7, 9};
  const auto got = run_2d(residual, psf, 8, 8, peak, 1.0f);
  const auto expected = host_subtract(residual, kH, kW, psf, 8, 8, peak, 1.0f);
  expect_matches_oracle(got, expected);
}

TEST_F(SubtractComponent, MultiFrequencyOverloadUsesPerChannelCoeffs)
{
  const int n_freq = 3, ph = 5, pw = 5;
  std::mt19937 rng(83);
  std::vector<float> residual(n_freq * kH * kW);
  fdtest::fill_uniform(rng, residual, -1.0f, 1.0f);
  std::vector<float> psf(n_freq * ph * pw);
  fdtest::fill_uniform(rng, psf, 0.0f, 1.0f);
  const std::vector<float> coeffs = {1.0f, -0.5f, 0.25f};
  const float gain = 0.4f;

  const auto sr = res().make_stream();

  for (const auto& peak : {std::pair<int, int>{8, 10}, {0, 0}, {kH - 1, kW - 1}}) {
    fdtest::device_buffer<float> d_res(res(), sr, residual);
    fdtest::device_buffer<float> d_psf(res(), sr, psf);
    fdtest::device_buffer<float> d_coeffs(res(), sr, coeffs);

    core::device_span3d<float> res_view(d_res.get(), n_freq, kH, kW);
    core::device_span3d<float> psf_view(d_psf.get(), n_freq, ph, pw);
    core::device_vect<float> coeffs_view(d_coeffs.get(), n_freq);

    common::subtract_component_async(sr, res_view, psf_view, coeffs_view, peak, gain);
    sr.sync();

    const auto got = d_res.to_host();
    for (int f = 0; f < n_freq; ++f) {
      const std::vector<float> res_f(residual.begin() + f * kH * kW, residual.begin() + (f + 1) * kH * kW);
      const std::vector<float> psf_f(psf.begin() + f * ph * pw, psf.begin() + (f + 1) * ph * pw);
      const auto expected = host_subtract(res_f, kH, kW, psf_f, ph, pw, peak, gain, coeffs.at(f));
      for (int i = 0; i < kH * kW; ++i)
        ASSERT_FLOAT_EQ(got.at(f * kH * kW + i), expected.at(i))
            << "freq " << f << " flat index " << i << " peak (" << peak.first << ", " << peak.second << ")";
    }
  }
}
