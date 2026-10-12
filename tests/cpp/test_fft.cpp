#include <gtest/gtest.h>

#include <algorithm>
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
// next_fast_size / compute_padding — pure host
// ============================================================================

namespace {

bool is_7_smooth(int n)
{
  for (int r : {2, 3, 5, 7})
    while (n % r == 0) n /= r;
  return n == 1;
}

}  // namespace

TEST(NextFastSize, ReturnsSmallest7SmoothAtLeastN)
{
  for (int n = 1; n <= 2000; ++n) {
    const int m = linalg::next_fast_size(n);
    ASSERT_GE(m, n);
    ASSERT_TRUE(is_7_smooth(m)) << "next_fast_size(" << n << ") = " << m << " is not 7-smooth";
    for (int k = n; k < m; ++k)
      ASSERT_FALSE(is_7_smooth(k)) << "next_fast_size(" << n << ") = " << m << " but " << k << " is 7-smooth";
  }
}

// ============================================================================
// pad_async / crop_async: pure data movement, compared element-exact against a
// host replica of the layout.
// ============================================================================

// CUDA only: the host backend has no FFT route.
#ifdef FAST_DECONV_WITH_CUDA

namespace {

// Host oracle for pad_async: each (nx, ny) image at the top-left of a zero (px, py) plane.
std::vector<float> host_pad(const std::vector<float>& in, int n_batch, int nx, int ny, int px, int py)
{
  std::vector<float> out(static_cast<std::size_t>(n_batch) * px * py, 0.0f);
  for (int b = 0; b < n_batch; ++b)
    for (int r = 0; r < nx; ++r)
      for (int c = 0; c < ny; ++c)
        out.at(static_cast<std::size_t>(b) * px * py + flat(r, c, py)) =
            in.at(static_cast<std::size_t>(b) * nx * ny + flat(r, c, ny));
  return out;
}

// The layout kernels only read the geometry, so set it explicitly: these cases
// pick padded sizes the gap constructor would not produce.
linalg::fft_dims explicit_dims(int nx, int ny, int px, int py)
{
  linalg::fft_dims d;
  d.input_nrow = nx;
  d.input_ncol = ny;
  d.padded_nrow = px;
  d.padded_ncol = py;
  d.freq_nrow = px;
  d.freq_ncol = py / 2 + 1;
  return d;
}

// Distinct, order-revealing values so any permutation of the layout shows up.
std::vector<float> iota_image(int n, float offset = 0.0f)
{
  std::vector<float> img(n);
  for (int i = 0; i < n; ++i) img.at(i) = offset + static_cast<float>(i + 1);
  return img;
}

// (nx, ny) -> (px, py) cases: odd and even gaps, and no padding at all.
struct layout_case {
  int nx, ny, px, py;
};
constexpr layout_case kLayoutCases[] = {{5, 6, 8, 9}, {6, 6, 10, 12}, {7, 5, 7, 5}};

}  // namespace
#endif

class FftLayout : public fdtest::BackendTest {};

// Gaussian(0) is 1 everywhere, so a zero-sigma convolution through the padded grid
// must give the input back: pad, R2C, the 1/padded_total normalization, C2R and
// crop all have to agree. Running it twice from the same spectrum checks that
// convolve leaves the spectrum intact, as the per-scale search relies on.
TEST_F(FftLayout, ZeroSigmaConvolutionIsIdentityAndKeepsTheSpectrum)
{
  const auto sr = res().make_ctx();
  for (const auto& [rows, cols] : {std::pair{8, 10}, std::pair{9, 15}, std::pair{10, 9}}) {
    for (int batch : {1, 3, 9}) {
      SCOPED_TRACE(::testing::Message() << rows << 'x' << cols << " batch=" << batch);
      const linalg::gaussian_convolution_ctx conv(sr, batch, rows, cols, /*gap=*/5);
      const auto count = static_cast<std::size_t>(batch) * rows * cols;

      std::vector<float> input(count);
      for (std::size_t i = 0; i < count; ++i) input[i] = static_cast<float>((i * 17) % 101) / 101.0f - 0.5f;
      fdtest::device_buffer<float> d_input(sr, input), d_output(sr, count);

      auto spectrum = conv.make_spectrum();
      conv.forward(core::span3d<const float>(d_input.get(), batch, rows, cols), spectrum);
      for (int rep = 0; rep < 2; ++rep) {
        conv.convolve(spectrum, 0.0f, core::span3d<float>(d_output.get(), batch, rows, cols));
        const auto actual = d_output.to_host();
        for (std::size_t i = 0; i < count; ++i)
          ASSERT_NEAR(actual[i], input[i], 2e-5f) << "rep=" << rep << " index=" << i;
      }
    }
  }
}

// convolve, from a spectrum and in one shot, is the linear convolution (zero outside the image) with the
// band-limited Gaussian, in every plane of a batch, on odd and even, non-square sizes. The gap is the largest
// reach over the sigmas, so both backends are free of wrap-around: the FFT path pads the plane by it, the host
// each line by the reach of the sigma at hand.
TEST_F(FftLayout, ConvolveSpectrumMatchesLinearConvolutionOracle)
{
  const auto sr = res().make_ctx();
  const std::vector<float> sigmas{1.5f, 2.5f, 4.0f};
  for (const auto& [rows, cols] : {std::pair{24, 40}, std::pair{33, 21}}) {
    for (int batch : {1, 3}) {
      const linalg::gaussian_convolution_ctx conv(sr, batch, rows, cols, linalg::max_gaussian_reach(sigmas));
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

// psf_convolution entry against its definition, computed independently in double: conv is each channel
// convolved once, conv2 the channel-weighted mean of each channel convolved twice. The build takes the
// weighted mean first and convolves once with sigma * sqrt(2); this checks that shortcut. psf_convolution
// sizes its gaps from the reach of its sigmas, so the test holds on both backends.
TEST_F(FftLayout, PsfConvolutionMatchesPerChannelOracle)
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

// With a gap below the kernel's reach the convolution wraps. Both backends must wrap like a circular
// convolution of length next_fast_size(n + gap) per axis: CUDA through its FFT size, the host by capping
// its line padding at the gap.
TEST_F(FftLayout, ConvolveWithShortGapWrapsLikeCircularOracle)
{
  const auto sr = res().make_ctx();
  constexpr int rows = 20, cols = 27, batch = 2, gap = 3;
  const float sigma = 4.0f;  // reach ~23 px, far beyond the gap
  const auto count = static_cast<std::size_t>(batch) * rows * cols;

  std::vector<float> input(count);
  for (std::size_t i = 0; i < count; ++i) input[i] = static_cast<float>((i * 37) % 101) / 101.0f - 0.5f;
  fdtest::device_buffer<float> d_input(sr, input), d_output(sr, count);

  const linalg::gaussian_convolution_ctx conv(sr, batch, rows, cols, gap);
  conv.convolve(core::span3d<const float>(d_input.get(), batch, rows, cols), sigma,
                core::span3d<float>(d_output.get(), batch, rows, cols));
  const auto actual = d_output.to_host();

  const auto hr = fdtest::circular_gaussian_kernel(sigma, linalg::next_fast_size(rows + gap));
  const auto hc = fdtest::circular_gaussian_kernel(sigma, linalg::next_fast_size(cols + gap));
  const auto expected = fdtest::separable_circular_convolve(input, batch, rows, cols, hr, hc);
  for (std::size_t i = 0; i < count; ++i) ASSERT_NEAR(actual[i], expected[i], 2e-5f) << "index=" << i;

  // The wrap must be visible, or the test would not tell the two layouts apart.
  const auto linear = fdtest::separable_linear_convolve(
      input, batch, rows, cols, fdtest::band_limited_gaussian_taps(sigma, std::max(rows, cols)));
  float max_wrap = 0.0f;
  for (std::size_t i = 0; i < count; ++i) max_wrap = std::max(max_wrap, std::abs(expected[i] - linear[i]));
  EXPECT_GT(max_wrap, 1e-3f);
}

#ifdef FAST_DECONV_WITH_CUDA
TEST_F(FftLayout, PadMatchesHostOracle)
{
  const auto sr = res().make_ctx();
  constexpr int n_batch = 3;
  for (const auto& cs : kLayoutCases) {
    SCOPED_TRACE(::testing::Message() << cs.nx << "x" << cs.ny << " -> " << cs.px << "x" << cs.py);
    // Distinct slices so cross-batch mixups are visible.
    std::vector<float> in;
    for (int b = 0; b < n_batch; ++b) {
      const auto slice = iota_image(cs.nx * cs.ny, 100.0f * b);
      in.insert(in.end(), slice.begin(), slice.end());
    }
    fdtest::device_buffer<float> d_in(sr, in);
    fdtest::device_buffer<float> d_out(sr, static_cast<std::size_t>(n_batch) * cs.px * cs.py);

    linalg::pad_async(sr, explicit_dims(cs.nx, cs.ny, cs.px, cs.py), d_in.get(), d_out.get(), n_batch);
    sr.wait();

    const auto out = d_out.to_host();
    const auto expected = host_pad(in, n_batch, cs.nx, cs.ny, cs.px, cs.py);
    for (std::size_t i = 0; i < expected.size(); ++i) ASSERT_FLOAT_EQ(out.at(i), expected.at(i)) << "flat index " << i;
  }
}

TEST_F(FftLayout, PadThenCropRoundTripIsIdentity)
{
  const auto sr = res().make_ctx();
  constexpr int n_batch = 2;
  for (const auto& cs : kLayoutCases) {
    SCOPED_TRACE(::testing::Message() << cs.nx << "x" << cs.ny << " -> " << cs.px << "x" << cs.py);
    const auto in = iota_image(n_batch * cs.nx * cs.ny);
    fdtest::device_buffer<float> d_in(sr, in);
    fdtest::device_buffer<float> d_pad(sr, static_cast<std::size_t>(n_batch) * cs.px * cs.py);
    fdtest::device_buffer<float> d_back(sr, in.size());

    const auto dims = explicit_dims(cs.nx, cs.ny, cs.px, cs.py);
    linalg::pad_async(sr, dims, d_in.get(), d_pad.get(), n_batch);
    linalg::crop_async(sr, dims, d_pad.get(), d_back.get(), n_batch);
    sr.wait();

    const auto back = d_back.to_host();
    for (std::size_t i = 0; i < in.size(); ++i) ASSERT_FLOAT_EQ(back.at(i), in.at(i)) << "flat index " << i;
  }
}
#endif
