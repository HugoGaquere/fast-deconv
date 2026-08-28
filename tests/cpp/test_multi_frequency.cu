#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <cmath>
#include <fast_deconv/common/multi_frequency.hpp>
#include <fast_deconv/core/memory_types.hpp>
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

// Host double-precision replica of fit_coefficients, for any order.
//   SAX[f, o]  = xdes[f, o] * sqrt(jn[f])
//   A[f, o]    = SAX[f, o] * sqrt(w[f])
//   compact    = inv(A^T A) @ A^T @ (sqrt(w) * y)
//   per_chan   = SAX @ compact
// then the band-weighted-mean constraint w . per_chan == w . y is imposed:
//   z = inv(G) @ c  with c = SAX^T w,  lambda = (w.y - w.per_chan) / (c . z)
//   compact += lambda * z,  per_chan += lambda * SAX @ z
// The constraint is what makes the minor cycle's bookkeeping exact: the mean
// residual at the peak must drop by exactly gain * peak (see PreservesBandMean).
struct fit_oracle {
  std::vector<double> compact;
  std::vector<double> per_chan;
};

// Gauss-Jordan solve of g @ out = rhs for a small symmetric positive-definite g.
std::vector<double> solve_spd(std::vector<double> g, std::vector<double> rhs, int n)
{
  for (int col = 0; col < n; ++col) {
    int piv = col;
    for (int r = col + 1; r < n; ++r)
      if (std::abs(g.at(r * n + col)) > std::abs(g.at(piv * n + col))) piv = r;
    for (int c = 0; c < n; ++c) std::swap(g.at(col * n + c), g.at(piv * n + c));
    std::swap(rhs.at(col), rhs.at(piv));

    const double d = g.at(col * n + col);
    for (int c = 0; c < n; ++c) g.at(col * n + c) /= d;
    rhs.at(col) /= d;

    for (int r = 0; r < n; ++r) {
      if (r == col) continue;
      const double factor = g.at(r * n + col);
      for (int c = 0; c < n; ++c) g.at(r * n + c) -= factor * g.at(col * n + c);
      rhs.at(r) -= factor * rhs.at(col);
    }
  }
  return rhs;
}

fit_oracle host_fit(const std::vector<double>& xdes, const std::vector<double>& jn_at_peak,
                    const std::vector<double>& w, const std::vector<double>& y_at_peak, int n_order)
{
  const int n_freq = static_cast<int>(w.size());

  std::vector<double> sax(n_freq * n_order);
  for (int f = 0; f < n_freq; ++f)
    for (int o = 0; o < n_order; ++o) sax.at(f * n_order + o) = xdes.at(f * n_order + o) * std::sqrt(jn_at_peak.at(f));

  // G = SAX^T W SAX, aty = SAX^T W y, c = SAX^T w
  std::vector<double> g(n_order * n_order, 0.0);
  std::vector<double> aty(n_order, 0.0);
  std::vector<double> c(n_order, 0.0);
  for (int f = 0; f < n_freq; ++f) {
    for (int i = 0; i < n_order; ++i) {
      const double swi = sax.at(f * n_order + i) * w.at(f);
      for (int j = 0; j < n_order; ++j) g.at(i * n_order + j) += swi * sax.at(f * n_order + j);
      aty.at(i) += swi * y_at_peak.at(f);
      c.at(i) += swi;
    }
  }

  fit_oracle out;
  out.compact = solve_spd(g, aty, n_order);
  const std::vector<double> z = solve_spd(g, c, n_order);

  auto eval = [&](const std::vector<double>& coeffs) {
    std::vector<double> per_chan(n_freq, 0.0);
    for (int f = 0; f < n_freq; ++f)
      for (int o = 0; o < n_order; ++o) per_chan.at(f) += sax.at(f * n_order + o) * coeffs.at(o);
    return per_chan;
  };
  auto wdot = [&](const std::vector<double>& v) {
    double acc = 0.0;
    for (int f = 0; f < n_freq; ++f) acc += w.at(f) * v.at(f);
    return acc;
  };

  const std::vector<double> u = eval(z);
  const double denom = wdot(u);
  out.per_chan = eval(out.compact);
  const double lambda = (wdot(y_at_peak) - wdot(out.per_chan)) / denom;
  for (int o = 0; o < n_order; ++o) out.compact.at(o) += lambda * z.at(o);
  for (int f = 0; f < n_freq; ++f) out.per_chan.at(f) += lambda * u.at(f);
  return out;
}

// Host double-precision minimum-norm fit, for n_order >= n_freq. There are fewer bands
// than coefficients, so SAX theta = y has infinitely many solutions and the Gram matrix
// SAX^T W SAX is singular. Take the smallest solution instead, as DDFacet does:
//   H = SAX SAX^T (n_freq x n_freq),  theta = SAX^T H^-1 y
// The weights cancel exactly in this branch, so they are not an argument.
std::vector<double> host_fit_min_norm(const std::vector<double>& xdes, const std::vector<double>& jn_at_peak,
                                      const std::vector<double>& y_at_peak, int n_order)
{
  const int n_freq = static_cast<int>(jn_at_peak.size());

  std::vector<double> sax(n_freq * n_order);
  for (int f = 0; f < n_freq; ++f)
    for (int o = 0; o < n_order; ++o) sax.at(f * n_order + o) = xdes.at(f * n_order + o) * std::sqrt(jn_at_peak.at(f));

  std::vector<double> h(n_freq * n_freq, 0.0);
  for (int a = 0; a < n_freq; ++a)
    for (int b = 0; b < n_freq; ++b)
      for (int o = 0; o < n_order; ++o) h.at(a * n_freq + b) += sax.at(a * n_order + o) * sax.at(b * n_order + o);

  const std::vector<double> u = solve_spd(h, y_at_peak, n_freq);
  std::vector<double> theta(n_order, 0.0);
  for (int o = 0; o < n_order; ++o)
    for (int f = 0; f < n_freq; ++f) theta.at(o) += sax.at(f * n_order + o) * u.at(f);
  return theta;
}

}  // namespace

class FitCoefficients : public fdtest::GpuTest {
 protected:
  // Upload the scene and run fit_coefficients at @p peak. jones_norm is
  // constant per channel (value jn[f] at every pixel).
  std::pair<std::vector<float>, std::vector<float>> run(const std::vector<float>& xdes, const std::vector<float>& jn,
                                                        const std::vector<float>& weights,
                                                        const std::vector<float>& dirty, std::pair<int, int> peak,
                                                        int n_order = kOrder)
  {
    const int n_freq = static_cast<int>(weights.size());

    std::vector<float> jn_image(n_freq * kNpix);
    for (int f = 0; f < n_freq; ++f)
      for (int i = 0; i < kNpix; ++i) jn_image.at(f * kNpix + i) = jn.at(f);

    const auto sr = res().make_ctx();
    fdtest::device_buffer<float> d_dirty(res(), sr, dirty);
    fdtest::device_buffer<float> d_jn(res(), sr, jn_image);
    fdtest::device_buffer<float> d_w(res(), sr, weights);
    fdtest::device_buffer<float> d_xdes(res(), sr, xdes);
    fdtest::device_buffer<float> d_compact(res(), sr, static_cast<std::size_t>(n_order));
    fdtest::device_buffer<float> d_per_chan(res(), sr, static_cast<std::size_t>(n_freq));

    core::device_span3d<float> dirty_view(d_dirty.get(), n_freq, kNrow, kNcol);
    core::device_span3d<float> jn_view(d_jn.get(), n_freq, kNrow, kNcol);
    core::span1d<float> w_view(d_w.get(), n_freq);
    core::device_span2d<float> xdes_view(d_xdes.get(), n_freq, n_order);
    core::span1d<float> compact_view(d_compact.get(), n_order);
    core::span1d<float> per_chan_view(d_per_chan.get(), n_freq);

    mfs::fit_coefficients(sr, dirty_view, jn_view, w_view, xdes_view, peak, compact_view, per_chan_view);
    sr.wait();

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
  const auto expected = host_fit(xdes_d, jn, w, y, kOrder);

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
  const auto expected = host_fit(xdes_d, std::vector<double>(n_freq, 1.0), std::vector<double>(n_freq, 1.0), y, kOrder);

  for (int o = 0; o < kOrder; ++o)
    EXPECT_NEAR(compact.at(o), expected.compact.at(o), 1e-3 * std::abs(expected.compact.at(o)) + 1e-4) << "order " << o;
  for (int f = 0; f < n_freq; ++f)
    EXPECT_NEAR(per_chan.at(f), expected.per_chan.at(f), 1e-3 * std::abs(expected.per_chan.at(f)) + 1e-4)
        << "freq " << f;
}

// The minor cycle subtracts gain * per_chan from the residual cube and books the
// mean-residual drop as gain * peak. Those agree only if the fitted model
// reproduces the band-weighted mean at the peak, so pin that invariant directly.
// Steep jones norms make the unconstrained fit violate it without bound: these
// are the real per-band values at pixel (411,434) of the DDMSC test field, where
// the beam spans 6450x across the 5 bands and the run diverged.
TEST_F(FitCoefficients, PreservesBandWeightedMeanAtSteepJones)
{
  constexpr int kFitOrder = 3;
  const int n_freq = 5;
  const std::vector<double> nu = {1090.0, 1285.0, 1480.0, 1675.0, 1870.0};
  const std::vector<double> jn = {6.0591e-02, 1.4545e-02, 1.7321e-03, 6.0919e-05, 9.3927e-06};
  const std::vector<double> w = {0.2, 0.2, 0.2, 0.2, 0.2};
  const std::vector<double> y = {0.02, -0.015, 0.01, -0.005, 0.002};

  const double nu0 = 1480.0;
  std::vector<double> xdes_d(n_freq * kFitOrder);
  std::vector<float> xdes(n_freq * kFitOrder);
  std::vector<float> dirty(n_freq * kNpix, 0.0f);
  const std::pair<int, int> peak{1, 2};
  for (int f = 0; f < n_freq; ++f) {
    for (int o = 0; o < kFitOrder; ++o) {
      xdes_d.at(f * kFitOrder + o) = std::pow(nu.at(f) / nu0, o);
      xdes.at(f * kFitOrder + o) = static_cast<float>(xdes_d.at(f * kFitOrder + o));
    }
    dirty.at(f * kNpix + flat(peak.first, peak.second, kNcol)) = static_cast<float>(y.at(f));
  }

  const std::vector<float> jn_f(jn.begin(), jn.end());
  const std::vector<float> w_f(w.begin(), w.end());
  const auto [compact, per_chan] = run(xdes, jn_f, w_f, dirty, peak, kFitOrder);
  const auto expected = host_fit(xdes_d, jn, w, y, kFitOrder);

  double mean_model = 0.0;
  double mean_data = 0.0;
  for (int f = 0; f < n_freq; ++f) {
    mean_model += w.at(f) * per_chan.at(f);
    mean_data += w.at(f) * y.at(f);
  }
  EXPECT_NEAR(mean_model, mean_data, 1e-6) << "band-weighted mean not preserved: CLEAN bookkeeping would be wrong";

  // Same tolerance as the well-conditioned cases: cond(G) ~ 3e5 here, which the
  // device solve absorbs because it runs in double.
  for (int o = 0; o < kFitOrder; ++o)
    EXPECT_NEAR(compact.at(o), expected.compact.at(o), 1e-3 * std::abs(expected.compact.at(o)) + 1e-4) << "order " << o;
  for (int f = 0; f < n_freq; ++f)
    EXPECT_NEAR(per_chan.at(f), expected.per_chan.at(f), 1e-3 * std::abs(expected.per_chan.at(f)) + 1e-4)
        << "freq " << f;
}

// Flat jones norms put the constant vector in the model span, so the constraint
// is already satisfied and must not perturb the fit.
TEST_F(FitCoefficients, ConstraintIsInactiveAtFlatJones)
{
  const int n_freq = 4;
  const std::vector<double> log_nu = {0.0, 0.1, 0.2, 0.3};
  const std::vector<double> jn(n_freq, 1.0767);
  const std::vector<double> w = {0.25, 0.25, 0.25, 0.25};
  const std::vector<double> y = {3.1, 2.7, 2.9, 2.4};

  std::vector<double> xdes_d(n_freq * kOrder);
  std::vector<float> xdes(n_freq * kOrder);
  std::vector<float> dirty(n_freq * kNpix, 0.0f);
  const std::pair<int, int> peak{3, 1};
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

  double mean_model = 0.0;
  double mean_data = 0.0;
  for (int f = 0; f < n_freq; ++f) {
    mean_model += w.at(f) * per_chan.at(f);
    mean_data += w.at(f) * y.at(f);
  }
  EXPECT_NEAR(mean_model, mean_data, 1e-5);
}

// Fewer bands than coefficients: SAX is 2x4, so SAX^T W SAX is 4x4 of rank 2 and the
// least-squares branch would invert a singular matrix. The fit must take the
// minimum-norm route instead. These are the real numbers from the LOFAR bigms run
// (2 bands, NumFreqBasisFuncs=4) whose model diverged: band centres 131.932/155.370
// MHz against nu0 = 143.650818 MHz, and the values one component actually subtracted.
TEST_F(FitCoefficients, MinimumNormFitWhenFewerBandsThanOrder)
{
  constexpr int kFitOrder = 4;
  const int n_freq = 2;
  const std::vector<double> nu = {131.932e6, 155.370e6};
  const std::vector<double> jn = {0.9, 0.8};
  const std::vector<double> w = {0.5, 0.5};
  const std::vector<double> y = {0.0193, -0.0989};

  const double nu0 = 143.650818e6;
  std::vector<double> xdes_d(n_freq * kFitOrder);
  std::vector<float> xdes(n_freq * kFitOrder);
  std::vector<float> dirty(n_freq * kNpix, 0.0f);
  const std::pair<int, int> peak{2, 4};
  for (int f = 0; f < n_freq; ++f) {
    for (int o = 0; o < kFitOrder; ++o) {
      xdes_d.at(f * kFitOrder + o) = std::pow(nu.at(f) / nu0, o);
      xdes.at(f * kFitOrder + o) = static_cast<float>(xdes_d.at(f * kFitOrder + o));
    }
    dirty.at(f * kNpix + flat(peak.first, peak.second, kNcol)) = static_cast<float>(y.at(f));
  }

  const std::vector<float> jn_f(jn.begin(), jn.end());
  const std::vector<float> w_f(w.begin(), w.end());
  const auto [compact, per_chan] = run(xdes, jn_f, w_f, dirty, peak, kFitOrder);
  const auto expected = host_fit_min_norm(xdes_d, jn, y, kFitOrder);

  // With at least as many coefficients as bands the fit interpolates every band, so the
  // band-weighted mean is preserved automatically and the constraint must not be applied.
  for (int f = 0; f < n_freq; ++f)
    EXPECT_NEAR(per_chan.at(f), y.at(f), 1e-6) << "band " << f << " not reproduced exactly";

  for (int o = 0; o < kFitOrder; ++o)
    EXPECT_NEAR(compact.at(o), expected.at(o), 1e-3 * std::abs(expected.at(o)) + 1e-4) << "order " << o;

  // Guards the regression directly: inverting the singular Gram matrix reproduced the
  // bands just as exactly but returned coefficients of order 1e3 (the run's worst
  // component was [-1358, 1781, 553, -960]), which then blew up at the degrid
  // frequencies. The minimum-norm coefficients for this data are of order 0.1.
  double norm_sq = 0.0;
  for (int o = 0; o < kFitOrder; ++o) norm_sq += compact.at(o) * compact.at(o);
  EXPECT_LT(std::sqrt(norm_sq), 1.0) << "coefficients are not minimum-norm: the null-space component is unbounded";
}

// The square boundary: n_order == n_freq has a unique exact solution, which is also the
// minimum-norm one. Pins that the branch switches on >= and not >.
TEST_F(FitCoefficients, ExactFitAtSquareSystem)
{
  constexpr int kFitOrder = 3;
  const int n_freq = 3;
  const std::vector<double> nu = {120.0e6, 144.0e6, 168.0e6};
  const std::vector<double> jn = {0.7, 1.0, 0.6};
  const std::vector<double> w = {1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0};
  const std::vector<double> y = {0.05, -0.02, 0.011};

  const double nu0 = 144.0e6;
  std::vector<double> xdes_d(n_freq * kFitOrder);
  std::vector<float> xdes(n_freq * kFitOrder);
  std::vector<float> dirty(n_freq * kNpix, 0.0f);
  const std::pair<int, int> peak{1, 3};
  for (int f = 0; f < n_freq; ++f) {
    for (int o = 0; o < kFitOrder; ++o) {
      xdes_d.at(f * kFitOrder + o) = std::pow(nu.at(f) / nu0, o);
      xdes.at(f * kFitOrder + o) = static_cast<float>(xdes_d.at(f * kFitOrder + o));
    }
    dirty.at(f * kNpix + flat(peak.first, peak.second, kNcol)) = static_cast<float>(y.at(f));
  }

  const std::vector<float> jn_f(jn.begin(), jn.end());
  const std::vector<float> w_f(w.begin(), w.end());
  const auto [compact, per_chan] = run(xdes, jn_f, w_f, dirty, peak, kFitOrder);
  const auto expected = host_fit_min_norm(xdes_d, jn, y, kFitOrder);

  for (int f = 0; f < n_freq; ++f) EXPECT_NEAR(per_chan.at(f), y.at(f), 1e-6) << "band " << f;
  for (int o = 0; o < kFitOrder; ++o)
    EXPECT_NEAR(compact.at(o), expected.at(o), 1e-3 * std::abs(expected.at(o)) + 1e-4) << "order " << o;
}
