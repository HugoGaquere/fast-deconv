#pragma once

#include <algorithm>

namespace fast_deconv::common {

/// A 2D index into a row-major array. Ordered row-then-column, matching span2d's extent order.
struct index2d {
  int row, col;
};

/// Bounding box of a mask's foreground, half-open: rows [rmin, rmax), columns [cmin, cmax).
struct roi {
  int rmin, rmax, cmin, cmax;
};

/// Intersection of a small array B centered on an index of a large array A.
struct overlap_region {
  int arow0, acol0, brow0, bcol0;  // Top-left in A and B
  int nrow, ncol;                  // Overlap size
  int lda, ldb;                    // A and B row stride in elements
};

/// Clamps B's footprint, centered at @p center in A, to A's bounds.
inline overlap_region compute_overlap_region(index2d center, int a_nrow, int a_ncol, int b_nrow, int b_ncol)
{
  const int brow_origin = center.row - b_nrow / 2;
  const int bcol_origin = center.col - b_ncol / 2;
  const int arow0 = std::max(brow_origin, 0);
  const int acol0 = std::max(bcol_origin, 0);
  const int arow1 = std::min(brow_origin + b_nrow, a_nrow);  // exclusive
  const int acol1 = std::min(bcol_origin + b_ncol, a_ncol);  // exclusive
  return {arow0, acol0, arow0 - brow_origin, acol0 - bcol_origin, arow1 - arow0, acol1 - acol0, a_ncol, b_ncol};
}

}  // namespace fast_deconv::common
