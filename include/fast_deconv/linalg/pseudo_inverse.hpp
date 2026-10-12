#pragma once

#include <fast_deconv/core/exec_ctx.hpp>

namespace fast_deconv::linalg {

// Compute the Moore-Penrose pseudo-inverse of a row-major matrix A [n_freq, n_order].
// A_pinv: col-major [n_order, n_freq] output (device), pre-allocated.
//
// Two code paths depending on shape:
//   overdetermined (n_freq >= n_order): A_pinv = inv(A^T A) @ A^T
//       G  = A^T A  [n_order, n_order]
//       A_pinv = inv(G) @ A^T
//   underdetermined (n_freq <  n_order): A_pinv = A^T @ inv(A A^T)
//       G  = A A^T  [n_freq,  n_freq]
//       A_pinv = A^T @ inv(G)
// The underdetermined path is required when n_order > n_freq since A^T A is then
// rank-deficient (rank <= n_freq in an n_order x n_order space) and matinvBatched
// fails with info != 0.
//
// cuBLAS sees row-major A as col-major A_cm = A^T with shape [n_order, n_freq].
void compute_pseudo_inverse(const core::exec_ctx& ctx, const float* d_A, float* d_A_pinv, int n_rows, int n_cols);

}  // namespace fast_deconv::linalg