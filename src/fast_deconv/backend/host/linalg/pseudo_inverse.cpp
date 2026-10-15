#include <cstdint>
#include <fast_deconv/core/profiler.hpp>
#include <fast_deconv/linalg/pseudo_inverse.hpp>
#include <vector>

#include "../detail/gauss_jordan.hpp"

namespace fast_deconv::linalg {

// A is row-major [n_rows, n_cols]; A_pinv is col-major [n_cols, n_rows], so
// element (o, f) lands at d_Apinv[f * n_cols + o]. Accumulation is in double —
// the Gram matrix is the same badly-conditioned one the spectral fit solves.
void compute_pseudo_inverse(const core::exec_ctx& ctx, const float* d_A, float* d_Apinv, std::int64_t n_rows,
                            std::int64_t n_cols)
{
  FD_PROFILE_FN();
  const auto A = [&](std::int64_t r, std::int64_t c) { return static_cast<double>(d_A[r * n_cols + c]); };

  const bool underdetermined = n_cols > n_rows;
  const std::int64_t g_dim = underdetermined ? n_rows : n_cols;
  const auto g_at = [&](std::int64_t r, std::int64_t c) { return core::narrow<std::size_t>(r * g_dim + c); };

  // Step 1: G = A^T A (overdetermined) or A A^T (underdetermined).
  std::vector<double> G(core::narrow<std::size_t>(g_dim * g_dim), 0.0);
  for (std::int64_t i = 0; i < g_dim; i++) {
    for (std::int64_t j = 0; j < g_dim; j++) {
      double acc = 0.0;
      if (underdetermined)
        for (std::int64_t c = 0; c < n_cols; c++) acc += A(i, c) * A(j, c);
      else
        for (std::int64_t r = 0; r < n_rows; r++) acc += A(r, i) * A(r, j);
      G.at(g_at(i, j)) = acc;
    }
  }

  // Step 2: Ginv, from the identity as right-hand sides.
  std::vector<double> Ginv(core::narrow<std::size_t>(g_dim * g_dim), 0.0);
  for (std::int64_t i = 0; i < g_dim; i++) Ginv.at(g_at(i, i)) = 1.0;
  detail::solve_gauss_jordan(G.data(), Ginv.data(), g_dim, g_dim);

  // Step 3: P = Ginv A^T (overdetermined) or A^T Ginv (underdetermined),
  //         written straight into the col-major output.
  for (std::int64_t o = 0; o < n_cols; o++) {
    for (std::int64_t f = 0; f < n_rows; f++) {
      double acc = 0.0;
      if (underdetermined)
        for (std::int64_t k = 0; k < n_rows; k++) acc += A(k, o) * Ginv.at(g_at(k, f));
      else
        for (std::int64_t k = 0; k < n_cols; k++) acc += Ginv.at(g_at(o, k)) * A(f, k);
      d_Apinv[f * n_cols + o] = static_cast<float>(acc);
    }
  }
}

}  // namespace fast_deconv::linalg
