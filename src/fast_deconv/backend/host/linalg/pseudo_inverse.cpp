#include <fast_deconv/linalg/pseudo_inverse.hpp>
#include <vector>

#include "../detail/gauss_jordan.hpp"

namespace fast_deconv::linalg {

// A is row-major [n_rows, n_cols]; A_pinv is col-major [n_cols, n_rows], so
// element (o, f) lands at d_Apinv[f * n_cols + o]. Accumulation is in double —
// the Gram matrix is the same badly-conditioned one the spectral fit solves.
void compute_pseudo_inverse(const core::exec_ctx& ctx, const float* d_A, float* d_Apinv, int n_rows, int n_cols)
{
  const auto A = [&](int r, int c) { return static_cast<double>(d_A[r * n_cols + c]); };

  const bool underdetermined = n_cols > n_rows;
  const int g_dim = underdetermined ? n_rows : n_cols;

  // Step 1: G = A^T A (overdetermined) or A A^T (underdetermined).
  std::vector<double> G(static_cast<std::size_t>(g_dim) * g_dim, 0.0);
  for (int i = 0; i < g_dim; i++) {
    for (int j = 0; j < g_dim; j++) {
      double acc = 0.0;
      if (underdetermined)
        for (int c = 0; c < n_cols; c++) acc += A(i, c) * A(j, c);
      else
        for (int r = 0; r < n_rows; r++) acc += A(r, i) * A(r, j);
      G.at(static_cast<std::size_t>(i) * g_dim + j) = acc;
    }
  }

  // Step 2: Ginv, from the identity as right-hand sides.
  std::vector<double> Ginv(static_cast<std::size_t>(g_dim) * g_dim, 0.0);
  for (int i = 0; i < g_dim; i++) Ginv.at(static_cast<std::size_t>(i) * g_dim + i) = 1.0;
  detail::solve_gauss_jordan(G.data(), Ginv.data(), g_dim, g_dim);

  // Step 3: P = Ginv A^T (overdetermined) or A^T Ginv (underdetermined),
  //         written straight into the col-major output.
  for (int o = 0; o < n_cols; o++) {
    for (int f = 0; f < n_rows; f++) {
      double acc = 0.0;
      if (underdetermined)
        for (int k = 0; k < n_rows; k++) acc += A(k, o) * Ginv.at(static_cast<std::size_t>(k) * g_dim + f);
      else
        for (int k = 0; k < n_cols; k++) acc += Ginv.at(static_cast<std::size_t>(o) * g_dim + k) * A(f, k);
      d_Apinv[f * n_cols + o] = static_cast<float>(acc);
    }
  }
}

}  // namespace fast_deconv::linalg
