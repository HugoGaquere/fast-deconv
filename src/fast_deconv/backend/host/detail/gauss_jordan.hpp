#pragma once

#include <cmath>
#include <cstdint>
#include <stdexcept>

namespace fast_deconv::detail {

/// Gauss-Jordan with partial pivoting. Solves m (n x n) against nrhs right-hand
/// sides held row-major in rhs (n x nrhs); both are overwritten, rhs with the
/// solutions. Port of the device solve in backend/cuda/common/multi_frequency.cu,
/// kept in double for the same reason: the spectral Gram matrix is badly
/// conditioned wherever the per-band beam is steep.
inline void solve_gauss_jordan(double* m, double* rhs, std::int64_t n, std::int64_t nrhs)
{
  for (std::int64_t col = 0; col < n; ++col) {
    std::int64_t piv = col;
    for (std::int64_t r = col + 1; r < n; ++r)
      if (std::fabs(m[r * n + col]) > std::fabs(m[piv * n + col])) piv = r;
    for (std::int64_t c = 0; c < n; ++c) {
      const double t = m[col * n + c];
      m[col * n + c] = m[piv * n + c];
      m[piv * n + c] = t;
    }
    for (std::int64_t k = 0; k < nrhs; ++k) {
      const double t = rhs[col * nrhs + k];
      rhs[col * nrhs + k] = rhs[piv * nrhs + k];
      rhs[piv * nrhs + k] = t;
    }

    // Partial pivoting already picked the largest magnitude, so a zero here means
    // the whole column is zero: dividing would leak NaNs to the caller silently.
    const double d = m[col * n + col];
    if (d == 0.0) throw std::domain_error("solve_gauss_jordan: singular matrix");
    for (std::int64_t c = 0; c < n; ++c) m[col * n + c] /= d;
    for (std::int64_t k = 0; k < nrhs; ++k) rhs[col * nrhs + k] /= d;

    for (std::int64_t r = 0; r < n; ++r) {
      if (r == col) continue;
      const double factor = m[r * n + col];
      for (std::int64_t c = 0; c < n; ++c) m[r * n + c] -= factor * m[col * n + c];
      for (std::int64_t k = 0; k < nrhs; ++k) rhs[r * nrhs + k] -= factor * rhs[col * nrhs + k];
    }
  }
}

}  // namespace fast_deconv::detail
