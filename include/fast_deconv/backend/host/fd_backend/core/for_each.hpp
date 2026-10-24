#pragma once

#include <cstdint>
#include <fast_deconv/core/exec_ctx.hpp>

namespace fast_deconv::core {

// Below this the fork costs more than the loop: a clean footprint can be a few rows.
inline constexpr std::int64_t kMinParallelForEach = 64 * 1024;

/// f(i) for every i in [0, n).
template <typename F>
void for_each_index(const exec_ctx& /*ctx*/, std::int64_t n, F f)
{
  // Serial outside OpenMP: even a parallel region whose if() is false enters the runtime and builds a team.
  if (n <= kMinParallelForEach) {
    for (std::int64_t i = 0; i < n; i++) f(i);
    return;
  }
#pragma omp parallel for firstprivate(f)
  for (std::int64_t i = 0; i < n; i++) f(i);
}

/// f(r, c) for every (r, c) in [0, nrow) x [0, ncol), c innermost so the column loop vectorizes.
template <typename F>
void for_each_index(const exec_ctx& /*ctx*/, std::int64_t nrow, std::int64_t ncol, F f)
{
  if (nrow * ncol <= kMinParallelForEach) {
    for (std::int64_t r = 0; r < nrow; r++)
      for (std::int64_t c = 0; c < ncol; c++) f(r, c);
    return;
  }
#pragma omp parallel for firstprivate(f)
  for (std::int64_t r = 0; r < nrow; r++)
    for (std::int64_t c = 0; c < ncol; c++) f(r, c);
}

/// f(d, r, c) for every (d, r, c) in [0, depth) x [0, nrow) x [0, ncol), c innermost.
template <typename F>
void for_each_index(const exec_ctx& /*ctx*/, std::int64_t depth, std::int64_t nrow, std::int64_t ncol, F f)
{
  if (depth * nrow * ncol <= kMinParallelForEach) {
    for (std::int64_t d = 0; d < depth; d++)
      for (std::int64_t r = 0; r < nrow; r++)
        for (std::int64_t c = 0; c < ncol; c++) f(d, r, c);
    return;
  }
  // Collapsed: depth alone is a handful of channels, too few to fill the pool.
#pragma omp parallel for collapse(2) firstprivate(f)
  for (std::int64_t d = 0; d < depth; d++)
    for (std::int64_t r = 0; r < nrow; r++)
      for (std::int64_t c = 0; c < ncol; c++) f(d, r, c);
}

}  // namespace fast_deconv::core
