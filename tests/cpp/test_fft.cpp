#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fast_deconv/algorithm/psf_convolution.hpp>
#include <fast_deconv/linalg/fft.hpp>
#include <fast_deconv/linalg/gaussian_convolution.hpp>
#include <utility>
#include <vector>

#include "helpers/backend_test.hpp"
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

// Taps h[0..R] of the band-limited Gaussian, the image-domain kernel of exp(-2 pi^2 sigma^2 f^2) for
// f in [-1/2, 1/2] (what the frequency-domain scale convolution applies on an infinite grid):
// h[x] = 2 * integral_0^{1/2} H(f) cos(2 pi f x) df, by composite Simpson. sigma == 0 gives a delta.
std::vector<double> band_limited_gaussian_taps(double sigma, int R)
{
  constexpr double pi = 3.14159265358979323846;
  constexpr int M = 4000;  // even; the integrand has at most R / 2 oscillations on [0, 1/2]
  std::vector<double> h(static_cast<std::size_t>(R) + 1);
  for (int x = 0; x <= R; ++x) {
    double acc = 0.0;
    for (int k = 0; k <= M; ++k) {
      const double f = 0.5 * k / M;
      const double w = (k == 0 || k == M) ? 1.0 : (k % 2 ? 4.0 : 2.0);
      acc += w * std::exp(-2.0 * pi * pi * sigma * sigma * f * f) * std::cos(2.0 * pi * f * x);
    }
    h.at(x) = 2.0 * acc * (0.5 / M) / 3.0;
  }
  return h;
}

// Linear (zero outside) separable convolution of each (nrow, ncol) plane of a batch with the symmetric
// taps h[0..R], h[-a] = h[a]: rows then columns, in double.
std::vector<float> separable_linear_convolve(const std::vector<float>& img, int batch, int nrow, int ncol,
                                                    const std::vector<double>& h)
{
  const int R = static_cast<int>(h.size()) - 1;
  const auto tap = [&](int d) { return std::abs(d) <= R ? h.at(std::abs(d)) : 0.0; };
  const std::size_t plane = static_cast<std::size_t>(nrow) * ncol;
  std::vector<double> tmp(img.size());
  std::vector<float> out(img.size());
  for (int b = 0; b < batch; ++b) {
    const std::size_t o = b * plane;
    for (int r = 0; r < nrow; ++r)
      for (int c = 0; c < ncol; ++c) {
        double acc = 0.0;
        for (int k = 0; k < ncol; ++k) acc += tap(c - k) * img.at(o + flat(r, k, ncol));
        tmp.at(o + flat(r, c, ncol)) = acc;
      }
    for (int r = 0; r < nrow; ++r)
      for (int c = 0; c < ncol; ++c) {
        double acc = 0.0;
        for (int k = 0; k < nrow; ++k) acc += tap(r - k) * tmp.at(o + flat(k, c, ncol));
        out.at(o + flat(r, c, ncol)) = static_cast<float>(acc);
      }
  }
  return out;
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
      std::vector<float> output(count);
      const auto d_input = sr.stage(core::host_span3d<const float>(input.data(), batch, rows, cols));
      const core::span3d<const float> in = d_input;  // a container also converts to a CUDA spectrum
      auto out = sr.stage(core::host_span3d<float>(output.data(), batch, rows, cols));
      const auto read_out = [&] {
        sr.unstage(out, output.data());
        sr.wait();
        return output;
      };
      auto spectrum = conv.make_spectrum();
      conv.forward(in, spectrum);

      for (float sigma : sigmas) {
        SCOPED_TRACE(::testing::Message() << rows << 'x' << cols << " batch=" << batch << " sigma=" << sigma);
        const auto taps = band_limited_gaussian_taps(sigma, std::max(rows, cols));
        const auto expected = separable_linear_convolve(input, batch, rows, cols, taps);

        conv.convolve(spectrum, sigma, out);
        const auto actual = read_out();
        for (std::size_t i = 0; i < count; ++i) ASSERT_NEAR(actual[i], expected[i], 2e-5f) << "index=" << i;

        conv.convolve(in, sigma, out);
        const auto one_shot = read_out();
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
  const auto d_psf = sr.stage(core::host_span4d<const float>(psf.data(), 1, n_freq, rows, cols));
  const auto d_weights = sr.stage(core::host_span1d<const float>(weights.data(), n_freq));

  fast_deconv::algorithm::psf_convolution cache(sr, d_psf, {0.0f, sigma}, d_weights, /*gamma=*/0.1f);
  const auto e = cache.get(/*scale=*/1, /*facet=*/0);

  std::vector<float> conv(psf.size()), conv2(static_cast<std::size_t>(rows) * cols);
  sr.copy(core::host_span3d<float>(conv.data(), n_freq, rows, cols), e.conv);
  sr.copy(core::host_span2d<float>(conv2.data(), rows, cols), e.conv2);
  sr.wait();

  // Convolve twice on a grid extended by the kernel length, then crop once: cropping between the two
  // convolutions would drop the part of the first result that spreads past the image and flows back.
  const int m = std::max(rows, cols);
  const int erows = rows + 2 * m, ecols = cols + 2 * m;
  const auto taps = band_limited_gaussian_taps(sigma, m);
  std::vector<float> ext(static_cast<std::size_t>(n_freq) * erows * ecols, 0.0f);
  for (int f = 0; f < n_freq; ++f)
    for (int r = 0; r < rows; ++r)
      for (int c = 0; c < cols; ++c)
        ext.at(static_cast<std::size_t>(f) * erows * ecols + flat(r + m, c + m, ecols)) =
            psf.at(static_cast<std::size_t>(f) * rows * cols + flat(r, c, cols));
  const auto once = separable_linear_convolve(ext, n_freq, erows, ecols, taps);
  const auto twice = separable_linear_convolve(once, n_freq, erows, ecols, taps);
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
