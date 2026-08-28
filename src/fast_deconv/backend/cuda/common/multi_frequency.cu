#include <cstdint>
#include <fast_deconv/common/multi_frequency.hpp>
#include <fast_deconv/util/cuda_macros.hpp>
#include <stdexcept>

namespace fast_deconv::kernel {

// Largest spectral order the in-kernel solve keeps in local storage. The order is
// the number of frequency basis functions, which is a handful in practice.
constexpr int SPECTRAL_MAX_ORDER = 8;

// Gauss-Jordan with partial pivoting. Solves m (n x n) against nrhs right-hand sides
// held row-major in rhs (n x nrhs); both are overwritten, rhs with the solutions.
__device__ void solve_gauss_jordan(double* m, double* rhs, int n, int nrhs)
{
  for (int col = 0; col < n; ++col) {
    int piv = col;
    for (int r = col + 1; r < n; ++r)
      if (fabs(m[r * n + col]) > fabs(m[piv * n + col])) piv = r;
    for (int c = 0; c < n; ++c) {
      const double t = m[col * n + c];
      m[col * n + c] = m[piv * n + c];
      m[piv * n + c] = t;
    }
    for (int k = 0; k < nrhs; ++k) {
      const double t = rhs[col * nrhs + k];
      rhs[col * nrhs + k] = rhs[piv * nrhs + k];
      rhs[piv * nrhs + k] = t;
    }

    const double d = m[col * n + col];
    for (int c = 0; c < n; ++c) m[col * n + c] /= d;
    for (int k = 0; k < nrhs; ++k) rhs[col * nrhs + k] /= d;

    for (int r = 0; r < n; ++r) {
      if (r == col) continue;
      const double factor = m[r * n + col];
      for (int c = 0; c < n; ++c) m[r * n + c] -= factor * m[col * n + c];
      for (int k = 0; k < nrhs; ++k) rhs[r * nrhs + k] -= factor * rhs[col * nrhs + k];
    }
  }
}

// Solve the weighted spectral fit at one pixel. SAX[f,o] = sqrt(jones_norm[f]) * xdes[f,o]
// maps intrinsic coefficients theta to the apparent per-band flux; the branch taken
// depends on whether the bands can determine theta.
//
// Overdetermined (n_freq > n_order) — a genuine least-squares fit, solved under the
// band-weighted-mean constraint  w . per_chan == w . y:
//
//   G      = SAX^T W SAX,  theta = G^-1 SAX^T W y,  z = G^-1 SAX^T w
//   lambda = (w.y - w.(SAX theta)) / (w.(SAX z))
//   theta' = theta + lambda * z
//
// The minor cycle subtracts gain * per_chan from the residual cube and books the
// mean-residual drop as gain * peak. Those agree only if the fitted model reproduces
// the band-weighted mean at the peak. The plain projection does so only when the
// constant vector lies in the span of SAX, i.e. when sqrt(jones_norm) is flat across
// bands. It is not, near the field edge, where the per-band beam can span four orders
// of magnitude — there the unfitted mean runs away and the minor cycle diverges.
//
// Exact fit (n_order >= n_freq) — fewer bands than coefficients, so SAX theta = y has
// infinitely many solutions and G (n_order x n_order, rank n_freq) is singular. Take
// the minimum-norm one instead, as DDFacet's ClassFrequencyMachine does:
//
//   H = SAX SAX^T  (n_freq x n_freq),  theta = SAX^T H^-1 y
//
// The weights cancel exactly here, and the constraint is skipped rather than merely
// evaluating to zero: the fit interpolates every band, so w . per_chan == w . y holds
// identically and lambda's denominator would be formed on the rank-deficient part.
// Without this branch the solve inverts a singular G, and the null-space component it
// invents is invisible in per_chan but corrupts the reported coefficients, which are
// later evaluated at the degrid frequencies.
//
// Everything is done in double. G is badly conditioned wherever the per-band beam is
// steep (cond ~ 3e5 at the edge of one L-band field, and it climbs fast with order),
// and in float32 the solve returns a model several times larger than the data it was
// fitting, which is itself enough to diverge the minor cycle. The systems are at most
// SPECTRAL_MAX_ORDER square, so double costs nothing here.
//
// One block, one thread: this is a tiny sequential solve, and the launch dominates.
__global__ void spectral_fit_kernel(float* compact_out, float* per_chan_out, const float* dirty, const float* xdes,
                                    const float* jones_norm, const float* weights, int n_freq, int n_order,
                                    int dirty_freq_stride, int dirty_peak_offset, int jn_freq_stride,
                                    int jn_peak_offset)
{
  if (threadIdx.x != 0 || blockIdx.x != 0) return;

  auto sqrt_jn = [&](int f) {
    return sqrt(static_cast<double>(jones_norm[static_cast<std::int64_t>(f) * jn_freq_stride + jn_peak_offset]));
  };
  auto y_at = [&](int f) {
    return static_cast<double>(dirty[static_cast<std::int64_t>(f) * dirty_freq_stride + dirty_peak_offset]);
  };

  double m[SPECTRAL_MAX_ORDER * SPECTRAL_MAX_ORDER] = {};
  double rhs[SPECTRAL_MAX_ORDER * 2] = {};
  double theta[SPECTRAL_MAX_ORDER] = {};

  if (n_order >= n_freq) {
    // H = SAX SAX^T, rhs = y. n_freq <= n_order <= SPECTRAL_MAX_ORDER, so both fit.
    for (int a = 0; a < n_freq; ++a) {
      const double sja = sqrt_jn(a);
      for (int b = 0; b < n_freq; ++b) {
        const double sjb = sqrt_jn(b);
        double acc = 0.0;
        for (int o = 0; o < n_order; ++o) acc += xdes[a * n_order + o] * sja * xdes[b * n_order + o] * sjb;
        m[a * n_freq + b] = acc;
      }
      rhs[a] = y_at(a);
    }
    solve_gauss_jordan(m, rhs, n_freq, 1);

    // theta = SAX^T u
    for (int o = 0; o < n_order; ++o) {
      double acc = 0.0;
      for (int f = 0; f < n_freq; ++f) acc += xdes[f * n_order + o] * sqrt_jn(f) * rhs[f];
      theta[o] = acc;
    }
  } else {
    // Normal equations, augmented with the constraint right-hand side:
    //   rhs[:, 0] = SAX^T W y      rhs[:, 1] = SAX^T w
    for (int f = 0; f < n_freq; ++f) {
      const double sj = sqrt_jn(f);
      const double wf = weights[f];
      const double yf = y_at(f);
      for (int i = 0; i < n_order; ++i) {
        const double swi = xdes[f * n_order + i] * sj * wf;
        for (int j = 0; j < n_order; ++j) m[i * n_order + j] += swi * xdes[f * n_order + j] * sj;
        rhs[i * 2] += swi * yf;
        rhs[i * 2 + 1] += swi;
      }
    }
    solve_gauss_jordan(m, rhs, n_order, 2);

    // lambda from the three band-weighted sums, then apply the correction.
    double mean_data = 0.0;
    double mean_model = 0.0;
    double denom = 0.0;
    for (int f = 0; f < n_freq; ++f) {
      const double sj = sqrt_jn(f);
      double pc = 0.0;
      double u = 0.0;
      for (int o = 0; o < n_order; ++o) {
        const double sax = xdes[f * n_order + o] * sj;
        pc += sax * rhs[o * 2];
        u += sax * rhs[o * 2 + 1];
      }
      mean_data += weights[f] * y_at(f);
      mean_model += weights[f] * pc;
      denom += weights[f] * u;
    }
    // denom == cos^2(1, span(SAX)) in the w metric, positive definite by construction.
    const double lambda = (mean_data - mean_model) / denom;

    for (int o = 0; o < n_order; ++o) theta[o] = rhs[o * 2] + lambda * rhs[o * 2 + 1];
  }

  for (int o = 0; o < n_order; ++o) compact_out[o] = static_cast<float>(theta[o]);
  for (int f = 0; f < n_freq; ++f) {
    const double sj = sqrt_jn(f);
    double acc = 0.0;
    for (int o = 0; o < n_order; ++o) acc += xdes[f * n_order + o] * sj * theta[o];
    per_chan_out[f] = static_cast<float>(acc);
  }
}

}  // namespace fast_deconv::kernel

namespace fast_deconv::multi_frequency {

void fit_coefficients(const core::exec_ctx& ctx, const core::span3d<float> residual,
                      const core::span3d<const float> jones_norm, const core::span1d<const float> weights_freq,
                      const core::span2d<float> xdes, const common::index2d peak_coords,
                      core::span1d<float> compact_coeffs_out, core::span1d<float> coeffs_per_chan_out)
{
  const int n_freq = xdes.extent(0);
  const int n_order = xdes.extent(1);
  const auto [peak_row, peak_col] = peak_coords;

  if (n_order > kernel::SPECTRAL_MAX_ORDER)
    throw std::invalid_argument("fit_coefficients: n_order exceeds SPECTRAL_MAX_ORDER");

  const int jn_freq_stride = jones_norm.extent(1) * jones_norm.extent(2);
  const int jn_peak_offset = peak_row * jones_norm.extent(2) + peak_col;
  const int dirty_freq_stride = residual.extent(1) * residual.extent(2);
  const int dirty_peak_offset = peak_row * residual.extent(2) + peak_col;

  kernel::spectral_fit_kernel<<<1, 1, 0, ctx.cuda_stream>>>(
      compact_coeffs_out.data_handle(), coeffs_per_chan_out.data_handle(), residual.data_handle(), xdes.data_handle(),
      jones_norm.data_handle(), weights_freq.data_handle(), n_freq, n_order, dirty_freq_stride, dirty_peak_offset,
      jn_freq_stride, jn_peak_offset);
  CHECK_LAST_CUDA_ERROR();
}

}  // namespace fast_deconv::multi_frequency
