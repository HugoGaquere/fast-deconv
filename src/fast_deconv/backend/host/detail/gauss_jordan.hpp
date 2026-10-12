#pragma once

#include <cmath>
#include <stdexcept>

namespace fast_deconv::detail {

/// Gauss-Jordan with partial pivoting. Solves m (n x n) against nrhs right-hand
/// sides held row-major in rhs (n x nrhs); both are overwritten, rhs with the
/// solutions. Port of the device solve in backend/cuda/common/multi_frequency.cu,
/// kept in double for the same reason: the spectral Gram matrix is badly
/// conditioned wherever the per-band beam is steep.
inline void solve_gauss_jordan(double* m, double* rhs, int n, int nrhs)
{
  for (int col = 0; col < n; ++col) {
    int piv = col;
    for (int r = col + 1; r < n; ++r)
      if (std::fabs(m[r * n + col]) > std::fabs(m[piv * n + col])) piv = r;
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

    // Partial pivoting already picked the largest magnitude, so a zero here means
    // the whole column is zero: dividing would leak NaNs to the caller silently.
    const double d = m[col * n + col];
    if (d == 0.0) throw std::domain_error("solve_gauss_jordan: singular matrix");
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

}  // namespace fast_deconv::detail
