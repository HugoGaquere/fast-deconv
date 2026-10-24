#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <fast_deconv/algorithm/psf_convolution.hpp>
#include <fast_deconv/linalg/fft.hpp>
#include <fast_deconv/linalg/gaussian_convolution.hpp>
#include <utility>
#include <vector>

#include "helpers/backend_test.hpp"
#include "helpers/device_buffers.hpp"
#include "helpers/host_oracles.hpp"

namespace core = fast_deconv::core;
namespace linalg = fast_deconv::linalg;
namespace fdtest = fast_deconv::test;

using fdtest::flat;

// ============================================================================
// next_fast_size — pure host
// ============================================================================

namespace {

bool is_7_smooth(std::int64_t n)
{
  for (std::int64_t r : {2, 3, 5, 7})
    while (n % r == 0) n /= r;
  return n == 1;
}

}  // namespace

TEST(NextFastSize, ReturnsSmallest7SmoothAtLeastN)
{
  for (std::int64_t n = 1; n <= 2000; ++n) {
    const std::int64_t m = linalg::next_fast_size(n);
    ASSERT_GE(m, n);
    ASSERT_TRUE(is_7_smooth(m)) << "next_fast_size(" << n << ") = " << m << " is not 7-smooth";
    for (std::int64_t k = n; k < m; ++k)
      ASSERT_FALSE(is_7_smooth(k)) << "next_fast_size(" << n << ") = " << m << " but " << k << " is 7-smooth";
  }
}

class GaussianConvolution : public fdtest::BackendTest {};

// convolve, from a precomputed spectrum and in one shot, is the linear convolution (zero outside the image,
// no wrap-around) with the band-limited Gaussian, in every plane of a batch, on odd, even and non-square
// sizes. Sigma 0 must give the input back. Every sigma reuses one spectrum, so convolve must leave it intact.
TEST_F(GaussianConvolution, MatchesLinearConvolutionOracle)
{
  const auto sr = res().make_ctx();
  const std::vector<float> sigmas{0.0f, 1.5f, 2.5f, 4.0f};
  for (const auto& [rows, cols] : {std::pair{24, 40}, std::pair{33, 21}}) {
    for (int batch : {1, 3}) {
      const linalg::gaussian_convolution_ctx conv(sr, batch, rows, cols, sigmas);
      const auto count = static_cast<std::size_t>(batch) * rows * cols;

      std::vector<float> input(count);
      for (std::size_t i = 0; i < count; ++i) input[i] = static_cast<float>((i * 37) % 101) / 101.0f - 0.5f;
      fdtest::device_buffer<float> d_input(sr, input), d_output(sr, count);
      const core::span3d<const float> in(d_input.get(), batch, rows, cols);
      auto spectrum = conv.make_spectrum();
      conv.forward(in, spectrum);

      for (float sigma : sigmas) {
        SCOPED_TRACE(::testing::Message() << rows << 'x' << cols << " batch=" << batch << " sigma=" << sigma);
        const core::span3d<float> out(d_output.get(), batch, rows, cols);
        const auto taps = fdtest::band_limited_gaussian_taps(sigma, std::max(rows, cols));
        const auto expected = fdtest::separable_linear_convolve(input, batch, rows, cols, taps);

        conv.convolve(spectrum, sigma, out);
        const auto actual = d_output.to_host();
        for (std::size_t i = 0; i < count; ++i) ASSERT_NEAR(actual[i], expected[i], 2e-5f) << "index=" << i;

        conv.convolve(in, sigma, out);
        const auto one_shot = d_output.to_host();
        for (std::size_t i = 0; i < count; ++i) ASSERT_NEAR(one_shot[i], expected[i], 2e-5f) << "one-shot index=" << i;
      }
    }
  }
}

// A psf_convolution entry against its definition: conv is each channel convolved once with the scale's
// Gaussian, conv2 the channel-weighted mean of each channel convolved twice. The cache takes the weighted
// mean first and convolves once with sigma * sqrt(2); this checks that shortcut.
TEST_F(GaussianConvolution, PsfConvolutionMatchesItsDefinition)
{
  const auto sr = res().make_ctx();
  constexpr int n_freq = 3, rows = 24, cols = 30;
  const float sigma = 2.0f;
  const std::vector<float> weights{0.5f, 0.3f, 0.2f};

  // One facet of peak-normalized Gaussian PSFs, a different width per channel.
  std::vector<float> psf;
  for (int f = 0; f < n_freq; ++f) {
    auto plane = fdtest::gaussian2d(rows, cols, 11.0, 14.0, 1.5 + 0.5 * f);
    const float peak = *std::max_element(plane.begin(), plane.end());
    for (float& v : plane) v /= peak;
    psf.insert(psf.end(), plane.begin(), plane.end());
  }
  fdtest::device_buffer<float> d_psf(sr, psf), d_weights(sr, weights);

  fast_deconv::algorithm::psf_convolution cache(sr, core::span4d<const float>(d_psf.get(), 1, n_freq, rows, cols),
                                                {0.0f, sigma}, core::span1d<const float>(d_weights.get(), n_freq),
                                                /*gamma=*/0.1f);
  const auto e = cache.get(/*scale=*/1, /*facet=*/0);

  std::vector<float> conv(psf.size()), conv2(static_cast<std::size_t>(rows) * cols);
  sr.copy_bytes(conv.data(), e.conv.data_handle(), conv.size() * sizeof(float));
  sr.copy_bytes(conv2.data(), e.conv2.data_handle(), conv2.size() * sizeof(float));
  sr.wait();

  // Convolve twice on a grid extended by the kernel length, then crop once: cropping between the two
  // convolutions would drop the part of the first result that spreads past the image and flows back.
  const int m = std::max(rows, cols);
  const int erows = rows + 2 * m, ecols = cols + 2 * m;
  const auto taps = fdtest::band_limited_gaussian_taps(sigma, m);
  std::vector<float> ext(static_cast<std::size_t>(n_freq) * erows * ecols, 0.0f);
  for (int f = 0; f < n_freq; ++f)
    for (int r = 0; r < rows; ++r)
      for (int c = 0; c < cols; ++c)
        ext.at(static_cast<std::size_t>(f) * erows * ecols + flat(r + m, c + m, ecols)) =
            psf.at(static_cast<std::size_t>(f) * rows * cols + flat(r, c, cols));
  const auto once = fdtest::separable_linear_convolve(ext, n_freq, erows, ecols, taps);
  const auto twice = fdtest::separable_linear_convolve(once, n_freq, erows, ecols, taps);
  const auto at = [&](const std::vector<float>& v, int f, int r, int c) {
    return v.at(static_cast<std::size_t>(f) * erows * ecols + flat(r + m, c + m, ecols));
  };
  for (int f = 0; f < n_freq; ++f)
    for (int r = 0; r < rows; ++r)
      for (int c = 0; c < cols; ++c)
        ASSERT_NEAR(conv.at(static_cast<std::size_t>(f) * rows * cols + flat(r, c, cols)), at(once, f, r, c), 2e-5f)
            << "conv f=" << f << " r=" << r << " c=" << c;
  for (int r = 0; r < rows; ++r)
    for (int c = 0; c < cols; ++c) {
      double mean = 0.0;
      for (int f = 0; f < n_freq; ++f) mean += weights.at(f) * at(twice, f, r, c);
      ASSERT_NEAR(conv2.at(flat(r, c, cols)), mean, 2e-5) << "conv2 r=" << r << " c=" << c;
    }
}
