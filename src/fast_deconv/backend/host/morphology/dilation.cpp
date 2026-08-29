#include <fast_deconv/morphology/dilation.hpp>

namespace fast_deconv::morphology {

void binary_dilation(const core::exec_ctx& ctx, core::span2d<bool> data, core::span2d<bool> structure,
                     common::roi structure_roi, core::span2d<bool> out)
{
  const int nrow = data.extent(0);
  const int ncol = data.extent(1);
  const int nrow_se = structure_roi.rmax - structure_roi.rmin;
  const int ncol_se = structure_roi.cmax - structure_roi.cmin;

  // True when the structuring element, centered on (r, c), covers a set pixel.
  const auto hits = [&](int r, int c) {
    const int row_start = r - nrow_se / 2;
    const int col_start = c - ncol_se / 2;

    for (int i = 0; i < nrow_se; i++) {
      const int rr = row_start + i;
      if (rr < 0 || rr >= nrow) continue;
      for (int j = 0; j < ncol_se; j++) {
        const int cc = col_start + j;
        if (cc < 0 || cc >= ncol) continue;
        if (data(rr, cc) && structure(structure_roi.rmin + i, structure_roi.cmin + j)) return true;
      }
    }
    return false;
  };

  for (int r = 0; r < nrow; r++) {
    for (int c = 0; c < ncol; c++) {
      out(r, c) = data(r, c) || hits(r, c);
    }
  }
}

}  // namespace fast_deconv::morphology
