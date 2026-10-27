#pragma once

#include <algorithm>
#include <cstdint>

namespace fast_deconv::core {

/// A 2D index into a row-major array. Ordered row-then-column, matching span2d's extent order.
struct index2d {
  std::int64_t row, col;
};

/// Bounding box of a mask's foreground, half-open: rows [rmin, rmax), columns [cmin, cmax).
struct roi {
  std::int64_t rmin, rmax, cmin, cmax;
};

/// Intersection of a small array B centered on an index of a large array A.
struct overlap_region {
  std::int64_t arow0, acol0, brow0, bcol0;  // Top-left in A and B
  std::int64_t nrow, ncol;                  // Overlap size
  std::int64_t lda, ldb;                    // A and B row stride in elements
};

/// Clamps B's footprint, centered at @p center in A, to A's bounds.
inline overlap_region compute_overlap_region(index2d center, std::int64_t a_nrow, std::int64_t a_ncol,
                                             std::int64_t b_nrow, std::int64_t b_ncol)
{
  const std::int64_t brow_origin = center.row - b_nrow / 2;
  const std::int64_t bcol_origin = center.col - b_ncol / 2;
  const std::int64_t arow0 = std::max<std::int64_t>(brow_origin, 0);
  const std::int64_t acol0 = std::max<std::int64_t>(bcol_origin, 0);
  const std::int64_t arow1 = std::min(brow_origin + b_nrow, a_nrow);  // exclusive
  const std::int64_t acol1 = std::min(bcol_origin + b_ncol, a_ncol);  // exclusive
  return {arow0, acol0, arow0 - brow_origin, acol0 - bcol_origin, arow1 - arow0, acol1 - acol0, a_ncol, b_ncol};
}

}  // namespace fast_deconv::core
