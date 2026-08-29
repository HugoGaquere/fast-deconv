#include <cmath>
#include <fast_deconv/common/multi_frequency.hpp>
#include <stdexcept>

#include "../detail/gauss_jordan.hpp"

namespace fast_deconv::multi_frequency {

namespace {
// Largest spectral order the solve keeps in local storage. The order is the
// number of frequency basis functions, which is a handful in practice.
constexpr int kSpectralMaxOrder = 8;
}  // namespace

// Port of spectral_fit_kernel (backend/cuda/common/multi_frequency.cu), which is
// already a one-thread sequential solve; see there for the derivation of the two
// branches and why everything is done in double.
void fit_coefficients(const core::exec_ctx& ctx, const core::span3d<float> residual,
                      const core::span3d<const float> jones_norm, const core::span1d<const float> weights_freq,
                      const core::span2d<float> xdes, const common::index2d peak_coords,
                      core::span1d<float> compact_coeffs_out, core::span1d<float> coeffs_per_chan_out)
{
  const int n_freq = xdes.extent(0);
  const int n_order = xdes.extent(1);
  const auto [peak_row, peak_col] = peak_coords;

  if (n_order > kSpectralMaxOrder) throw std::invalid_argument("fit_coefficients: n_order exceeds SPECTRAL_MAX_ORDER");

  const auto sqrt_jn = [&](int f) { return std::sqrt(static_cast<double>(jones_norm(f, peak_row, peak_col))); };
  const auto y_at = [&](int f) { return static_cast<double>(residual(f, peak_row, peak_col)); };

  double m[kSpectralMaxOrder * kSpectralMaxOrder] = {};
  double rhs[kSpectralMaxOrder * 2] = {};
  double theta[kSpectralMaxOrder] = {};

  if (n_order >= n_freq) {
    // Minimum-norm fit: H = SAX SAX^T, theta = SAX^T H^-1 y.
    // n_freq <= n_order <= kSpectralMaxOrder, so both fixed buffers fit.
    for (int a = 0; a < n_freq; ++a) {
      const double sja = sqrt_jn(a);
      for (int b = 0; b < n_freq; ++b) {
        const double sjb = sqrt_jn(b);
        double acc = 0.0;
        for (int o = 0; o < n_order; ++o) acc += xdes(a, o) * sja * xdes(b, o) * sjb;
        m[a * n_freq + b] = acc;
      }
      rhs[a] = y_at(a);
    }
    detail::solve_gauss_jordan(m, rhs, n_freq, 1);

    for (int o = 0; o < n_order; ++o) {
      double acc = 0.0;
      for (int f = 0; f < n_freq; ++f) acc += xdes(f, o) * sqrt_jn(f) * rhs[f];
      theta[o] = acc;
    }
  } else {
    // Normal equations, augmented with the constraint right-hand side:
    //   rhs[:, 0] = SAX^T W y      rhs[:, 1] = SAX^T w
    for (int f = 0; f < n_freq; ++f) {
      const double sj = sqrt_jn(f);
      const double wf = weights_freq(f);
      const double yf = y_at(f);
      for (int i = 0; i < n_order; ++i) {
        const double swi = xdes(f, i) * sj * wf;
        for (int j = 0; j < n_order; ++j) m[i * n_order + j] += swi * xdes(f, j) * sj;
        rhs[i * 2] += swi * yf;
        rhs[i * 2 + 1] += swi;
      }
    }
    detail::solve_gauss_jordan(m, rhs, n_order, 2);

    // lambda from the three band-weighted sums, then apply the correction.
    double mean_data = 0.0;
    double mean_model = 0.0;
    double denom = 0.0;
    for (int f = 0; f < n_freq; ++f) {
      const double sj = sqrt_jn(f);
      double pc = 0.0;
      double u = 0.0;
      for (int o = 0; o < n_order; ++o) {
        const double sax = xdes(f, o) * sj;
        pc += sax * rhs[o * 2];
        u += sax * rhs[o * 2 + 1];
      }
      mean_data += weights_freq(f) * y_at(f);
      mean_model += weights_freq(f) * pc;
      denom += weights_freq(f) * u;
    }
    // denom == cos^2(1, span(SAX)) in the w metric, positive definite by construction.
    const double lambda = (mean_data - mean_model) / denom;

    for (int o = 0; o < n_order; ++o) theta[o] = rhs[o * 2] + lambda * rhs[o * 2 + 1];
  }

  for (int o = 0; o < n_order; ++o) compact_coeffs_out(o) = static_cast<float>(theta[o]);
  for (int f = 0; f < n_freq; ++f) {
    const double sj = sqrt_jn(f);
    double acc = 0.0;
    for (int o = 0; o < n_order; ++o) acc += xdes(f, o) * sj * theta[o];
    coeffs_per_chan_out(f) = static_cast<float>(acc);
  }
}

}  // namespace fast_deconv::multi_frequency
