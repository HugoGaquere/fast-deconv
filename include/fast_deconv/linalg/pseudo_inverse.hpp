#pragma once

#include <cstdint>
#include <fast_deconv/core/exec_ctx.hpp>

namespace fast_deconv::linalg {

/**
 * @brief Moore-Penrose pseudo-inverse of a row-major matrix A.
 *
 * Two paths depending on shape:
 *   - overdetermined (n_rows >= n_cols): G = A^T A (n_cols, n_cols), A_pinv = inv(G) A^T
 *   - underdetermined (n_rows < n_cols): G = A A^T (n_rows, n_rows), A_pinv = A^T inv(G)
 *
 * The underdetermined path is required because A^T A is then rank-deficient (rank <= n_rows in an
 * n_cols x n_cols space) and cannot be inverted.
 *
 * @param[in]  ctx       Execution context.
 * @param[in]  d_A       A, backend memory, row-major (n_rows, n_cols).
 * @param[out] d_A_pinv  A_pinv, backend memory, col-major (n_cols, n_rows), pre-allocated.
 * @param[in]  n_rows    Rows of A (frequencies in the spectral fit).
 * @param[in]  n_cols    Columns of A (spectral order).
 */
void compute_pseudo_inverse(const core::exec_ctx& ctx, const float* d_A, float* d_A_pinv, std::int64_t n_rows,
                            std::int64_t n_cols);

}  // namespace fast_deconv::linalg