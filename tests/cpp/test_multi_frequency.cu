#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <cmath>
#include <fast_deconv/common/multi_frequency.hpp>
#include <fast_deconv/core/span_types.hpp>
#include <utility>
#include <vector>

#include "helpers/device_buffers.hpp"
#include "helpers/gpu_test.hpp"
#include "helpers/host_oracles.hpp"
#include "helpers/rng.hpp"

namespace core = fast_deconv::core;
namespace mfs = fast_deconv::multi_frequency;
namespace fdtest = fast_deconv::test;

using fdtest::flat;

namespace {

constexpr int kNrow = 5;
constexpr int kNcol = 7;
constexpr int kNpix = kNrow * kNcol;
constexpr int kOrder = 2;

// Host double-precision replica of fit_coefficients:
//   A[f, o]    = xdes[f, o] * sqrt(jn[f]) * sqrt(w[f])
//   compact    = pinv(A) @ (sqrt(w) * y)          (normal equations, kOrder = 2)
//   per_chan[f]= (xdes[f, :] * sqrt(jn[f])) @ compact
struct fit_oracle {
  std::vector<double> compact;
  std::vector<double> per_chan;
};

fit_oracle host_fit(const std::vector<double>& xdes, const std::vector<double>& jn_at_peak,
                    const std::vector<double>& w, const std::vector<double>& y_at_peak)
{
  const int n_freq = static_cast<int>(w.size());

  std::vector<double> a(n_freq * kOrder);
  for (int f = 0; f < n_freq; ++f)
    for (int o = 0; o < kOrder; ++o)
      a.at(f * kOrder + o) = xdes.at(f * kOrder + o) * std::sqrt(jn_at_peak.at(f)) * std::sqrt(w.at(f));

  // Normal equations with a 2x2 Gram matrix.
  double g[2][2] = {{0.0, 0.0}, {0.0, 0.0}};
  double aty[2] = {0.0, 0.0};
  for (int f = 0; f < n_freq; ++f) {
    const double wy = std::sqrt(w.at(f)) * y_at_peak.at(f);
    for (int i = 0; i < 2; ++i) {
      for (int j = 0; j < 2; ++j) g[i][j] += a.at(f * kOrder + i) * a.at(f * kOrder + j);
      aty[i] += a.at(f * kOrder + i) * wy;
    }
  }
  const double det = g[0][0] * g[1][1] - g[0][1] * g[1][0];

  fit_oracle out;
  out.compact = {(g[1][1] * aty[0] - g[0][1] * aty[1]) / det, (-g[1][0] * aty[0] + g[0][0] * aty[1]) / det};
  out.per_chan.resize(n_freq);
  for (int f = 0; f < n_freq; ++f) {
    double acc = 0.0;
    for (int o = 0; o < kOrder; ++o) acc += xdes.at(f * kOrder + o) * std::sqrt(jn_at_peak.at(f)) * out.compact.at(o);
    out.per_chan.at(f) = acc;
  }
  return out;
}

}  // namespace

class FitCoefficients : public fdtest::GpuTest {
 protected:
  // Upload the scene and run fit_coefficients at @p peak. jones_norm is
  // constant per channel (value jn[f] at every pixel).
  std::pair<std::vector<float>, std::vector<float>> run(const std::vector<float>& xdes, const std::vector<float>& jn,
                                                        const std::vector<float>& weights,
                                                        const std::vector<float>& dirty, std::pair<int, int> peak)
  {
    const int n_freq = static_cast<int>(weights.size());

    std::vector<float> jn_image(n_freq * kNpix);
    for (int f = 0; f < n_freq; ++f)
      for (int i = 0; i < kNpix; ++i) jn_image.at(f * kNpix + i) = jn.at(f);

    const auto sr = res().make_stream();
    fdtest::device_buffer<float> d_dirty(res(), sr, dirty);
    fdtest::device_buffer<float> d_jn(res(), sr, jn_image);
    fdtest::device_buffer<float> d_w(res(), sr, weights);
    fdtest::device_buffer<float> d_xdes(res(), sr, xdes);
    fdtest::device_buffer<float> d_compact(res(), sr, static_cast<std::size_t>(kOrder));
    fdtest::device_buffer<float> d_per_chan(res(), sr, static_cast<std::size_t>(n_freq));

    core::device_span3d<float> dirty_view(d_dirty.get(), n_freq, kNrow, kNcol);
    core::device_span3d<float> jn_view(d_jn.get(), n_freq, kNrow, kNcol);
    core::device_vect<float> w_view(d_w.get(), n_freq);
    core::device_span2d<float> xdes_view(d_xdes.get(), n_freq, kOrder);
    core::device_vect<float> compact_view(d_compact.get(), kOrder);
    core::device_vect<float> per_chan_view(d_per_chan.get(), n_freq);

    mfs::fit_coefficients(sr, dirty_view, jn_view, w_view, xdes_view, peak, compact_view, per_chan_view);
    sr.sync();

    return {d_compact.to_host(), d_per_chan.to_host()};
  }
};

// With unit jones/weights and dirty[f, peak] = xdes[f, :] @ alpha, the fit is
// exactly determined: compact must recover alpha and per_chan must reproduce
// the peak values.
TEST_F(FitCoefficients, RecoversExactCoefficientsWithUnitWeights)
{
  const int n_freq = 4;
  const std::vector<double> log_nu = {0.0, 0.1, 0.2, 0.3};
  const std::vector<double> alpha = {2.5, -1.2};

  std::vector<float> xdes(n_freq * kOrder);
  std::vector<float> dirty(n_freq * kNpix, 0.0f);
  const std::pair<int, int> peak{2, 3};
  for (int f = 0; f < n_freq; ++f) {
    xdes.at(f * kOrder + 0) = 1.0f;
    xdes.at(f * kOrder + 1) = static_cast<float>(log_nu.at(f));
    dirty.at(f * kNpix + flat(peak.first, peak.second, kNcol)) =
        static_cast<float>(alpha.at(0) + log_nu.at(f) * alpha.at(1));
  }

  const auto [compact, per_chan] =
      run(xdes, std::vector<float>(n_freq, 1.0f), std::vector<float>(n_freq, 1.0f), dirty, peak);

  EXPECT_NEAR(compact.at(0), alpha.at(0), 1e-3f);
  EXPECT_NEAR(compact.at(1), alpha.at(1), 1e-3f);
  for (int f = 0; f < n_freq; ++f)
    EXPECT_NEAR(per_chan.at(f), dirty.at(f * kNpix + flat(peak.first, peak.second, kNcol)), 1e-3f) << "freq " << f;
}

TEST_F(FitCoefficients, WeightedJonesFitMatchesHostLeastSquares)
{
  const int n_freq = 4;
  const std::vector<double> log_nu = {0.0, 0.1, 0.2, 0.3};
  const std::vector<double> jn = {1.0, 1.2, 0.9, 1.1};
  const std::vector<double> w = {0.5, 1.0, 0.75, 1.25};
  const std::vector<double> y = {3.1, 2.7, 2.9, 2.4};  // peak values, deliberately not an exact model fit

  std::vector<double> xdes_d(n_freq * kOrder);
  std::vector<float> xdes(n_freq * kOrder);
  std::vector<float> dirty(n_freq * kNpix, 0.1f);
  const std::pair<int, int> peak{4, 6};  // bottom-right corner pixel
  for (int f = 0; f < n_freq; ++f) {
    xdes_d.at(f * kOrder + 0) = 1.0;
    xdes_d.at(f * kOrder + 1) = log_nu.at(f);
    xdes.at(f * kOrder + 0) = 1.0f;
    xdes.at(f * kOrder + 1) = static_cast<float>(log_nu.at(f));
    dirty.at(f * kNpix + flat(peak.first, peak.second, kNcol)) = static_cast<float>(y.at(f));
  }

  const std::vector<float> jn_f(jn.begin(), jn.end());
  const std::vector<float> w_f(w.begin(), w.end());
  const auto [compact, per_chan] = run(xdes, jn_f, w_f, dirty, peak);
  const auto expected = host_fit(xdes_d, jn, w, y);

  for (int o = 0; o < kOrder; ++o)
    EXPECT_NEAR(compact.at(o), expected.compact.at(o), 1e-3 * std::abs(expected.compact.at(o)) + 1e-4) << "order " << o;
  for (int f = 0; f < n_freq; ++f)
    EXPECT_NEAR(per_chan.at(f), expected.per_chan.at(f), 1e-3 * std::abs(expected.per_chan.at(f)) + 1e-4)
        << "freq " << f;
}

TEST_F(FitCoefficients, OverdeterminedNoisyFitMatchesHostLeastSquares)
{
  const int n_freq = 6;
  const std::vector<double> log_nu = {0.0, 0.08, 0.15, 0.22, 0.29, 0.35};
  const std::vector<double> alpha = {1.8, -0.9};

  std::mt19937 rng(101);
  std::vector<double> y(n_freq);
  std::vector<double> xdes_d(n_freq * kOrder);
  std::vector<float> xdes(n_freq * kOrder);
  std::vector<float> dirty(n_freq * kNpix, 0.0f);
  const std::pair<int, int> peak{0, 0};  // top-left corner pixel
  for (int f = 0; f < n_freq; ++f) {
    y.at(f) = alpha.at(0) + log_nu.at(f) * alpha.at(1) + 0.05 * fdtest::normal01(rng);
    xdes_d.at(f * kOrder + 0) = 1.0;
    xdes_d.at(f * kOrder + 1) = log_nu.at(f);
    xdes.at(f * kOrder + 0) = 1.0f;
    xdes.at(f * kOrder + 1) = static_cast<float>(log_nu.at(f));
    dirty.at(f * kNpix + flat(peak.first, peak.second, kNcol)) = static_cast<float>(y.at(f));
  }

  const auto [compact, per_chan] =
      run(xdes, std::vector<float>(n_freq, 1.0f), std::vector<float>(n_freq, 1.0f), dirty, peak);
  const auto expected = host_fit(xdes_d, std::vector<double>(n_freq, 1.0), std::vector<double>(n_freq, 1.0), y);

  for (int o = 0; o < kOrder; ++o)
    EXPECT_NEAR(compact.at(o), expected.compact.at(o), 1e-3 * std::abs(expected.compact.at(o)) + 1e-4) << "order " << o;
  for (int f = 0; f < n_freq; ++f)
    EXPECT_NEAR(per_chan.at(f), expected.per_chan.at(f), 1e-3 * std::abs(expected.per_chan.at(f)) + 1e-4)
        << "freq " << f;
}
