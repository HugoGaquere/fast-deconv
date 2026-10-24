#include <gtest/gtest.h>

#include <fast_deconv/common/clean.hpp>
#include <fast_deconv/core/memory_types.hpp>
#include <random>
#include <vector>

#include "helpers/backend_test.hpp"
#include "helpers/device_buffers.hpp"
#include "helpers/host_oracles.hpp"
#include "helpers/rng.hpp"

namespace core = fast_deconv::core;
namespace common = fast_deconv::common;
namespace fdtest = fast_deconv::test;

using fdtest::flat;

namespace {

constexpr int kH = 16;
constexpr int kW = 20;  // non-square residual

// residual[peak + (i - ph/2, j - pw/2)] -= gain * psf[i, j], clipped to the image.
std::vector<float> host_subtract(std::vector<float> residual, const std::vector<float>& psf, int ph, int pw,
                                 common::index2d peak, float gain)
{
  for (int i = 0; i < ph; ++i)
    for (int j = 0; j < pw; ++j) {
      const int row = static_cast<int>(peak.row) - ph / 2 + i, col = static_cast<int>(peak.col) - pw / 2 + j;
      if (row >= 0 && row < kH && col >= 0 && col < kW)
        residual.at(flat(row, col, kW)) -= gain * psf.at(flat(i, j, pw));
    }
  return residual;
}

// Plane @p f of a (n_planes, plane) stack.
std::vector<float> plane_of(const std::vector<float>& v, int f, int plane)
{
  return {v.begin() + f * plane, v.begin() + (f + 1) * plane};
}

}  // namespace

class SubtractComponent : public fdtest::BackendTest {};

// The scaled PSF is subtracted around the peak and clipped at the image border; nothing else changes. The
// 8x8 PSF pins the centering: its pixel (4, 4) lands on the peak, 4 rows/cols before it and 3 after.
TEST_F(SubtractComponent, SubtractsPsfAroundPeakClippedAtBorders)
{
  std::mt19937 rng(71);
  std::vector<float> residual(kH * kW);
  fdtest::fill_uniform(rng, residual, -1.0f, 1.0f);
  const auto sr = res().make_ctx();

  for (const int n : {5, 8}) {
    std::vector<float> psf(n * n);
    fdtest::fill_uniform(rng, psf, 0.0f, 1.0f);

    for (const auto& peak : {common::index2d{8, 10}, {0, 0}, {0, kW - 1}, {kH - 1, 0}, {kH - 1, kW - 1}}) {
      SCOPED_TRACE(::testing::Message() << n << "x" << n << " psf, peak (" << peak.row << ", " << peak.col << ")");
      fdtest::device_buffer<float> d_res(sr, residual), d_psf(sr, psf);

      common::subtract_component(sr, core::span2d<float>(d_res.get(), kH, kW),
                                 core::span2d<const float>(d_psf.get(), n, n), peak, 0.5f);

      const auto got = d_res.to_host();
      const auto expected = host_subtract(residual, psf, n, n, peak, 0.5f);
      for (int i = 0; i < kH * kW; ++i) ASSERT_FLOAT_EQ(got.at(i), expected.at(i)) << "index " << i;
    }
  }
}

// The clean loop subtracts only where the residual still qualifies: masked pixels, and pixels that its
// criterion ranks below the threshold, keep their value. With abs, a negative pixel qualifies by magnitude.
TEST_F(SubtractComponent, SkipsMaskedPixelsAndPixelsBelowThreshold)
{
  std::vector<float> residual(kH * kW, 1.0f);
  std::vector<bool> mask(kH * kW, false);
  const int small = flat(7, 9, kW), negative = flat(8, 11, kW), masked = flat(9, 10, kW);
  residual.at(small) = 0.2f;
  residual.at(negative) = -0.8f;
  mask.at(masked) = true;

  const auto sr = res().make_ctx();
  fdtest::device_buffer<float> d_res(sr, residual), d_psf(sr, std::vector<float>(5 * 5, 1.0f));
  fdtest::device_buffer<bool> d_mask(sr, mask);

  common::subtract_component(sr, core::span2d<float>(d_res.get(), kH, kW), core::span2d<const float>(d_psf.get(), 5, 5),
                             {8, 10}, /*gain=*/0.5f, {.mask = d_mask.get(), .absolute = true}, /*threshold=*/0.5f);

  const auto got = d_res.to_host();
  EXPECT_FLOAT_EQ(got.at(flat(8, 10, kW)), 0.5f);  // in the window: 1 - 0.5
  EXPECT_FLOAT_EQ(got.at(small), 0.2f);            // |0.2| < 0.5: untouched
  EXPECT_FLOAT_EQ(got.at(negative), -1.3f);        // |-0.8| >= 0.5: subtracted
  EXPECT_FLOAT_EQ(got.at(masked), 1.0f);           // masked: untouched
  EXPECT_FLOAT_EQ(got.at(flat(0, 0, kW)), 1.0f);   // outside the window
}

// Every channel subtracts its own PSF scaled by gain * its spectral coefficient.
TEST_F(SubtractComponent, MultiFrequencyScalesEachChannelByItsCoeff)
{
  const int n_freq = 3, n = 5;
  std::mt19937 rng(83);
  std::vector<float> residual(n_freq * kH * kW), psf(n_freq * n * n);
  fdtest::fill_uniform(rng, residual, -1.0f, 1.0f);
  fdtest::fill_uniform(rng, psf, 0.0f, 1.0f);
  const std::vector<float> coeffs = {1.0f, -0.5f, 0.25f};
  const float gain = 0.4f;
  const auto sr = res().make_ctx();

  for (const auto& peak : {common::index2d{8, 10}, {kH - 1, kW - 1}}) {
    fdtest::device_buffer<float> d_res(sr, residual), d_psf(sr, psf), d_coeffs(sr, coeffs);

    common::subtract_component(sr, core::span3d<float>(d_res.get(), n_freq, kH, kW),
                               core::span3d<const float>(d_psf.get(), n_freq, n, n),
                               core::span1d<const float>(d_coeffs.get(), n_freq), peak, gain);

    const auto got = d_res.to_host();
    for (int f = 0; f < n_freq; ++f) {
      SCOPED_TRACE(::testing::Message() << "channel " << f << ", peak (" << peak.row << ", " << peak.col << ")");
      const auto expected =
          host_subtract(plane_of(residual, f, kH * kW), plane_of(psf, f, n * n), n, n, peak, gain * coeffs.at(f));
      for (int i = 0; i < kH * kW; ++i) ASSERT_FLOAT_EQ(got.at(f * kH * kW + i), expected.at(i)) << "index " << i;
    }
  }
}
