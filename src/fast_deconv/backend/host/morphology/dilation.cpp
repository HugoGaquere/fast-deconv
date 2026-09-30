#include <fast_deconv/core/profiler.hpp>
#include <fast_deconv/morphology/dilation.hpp>

namespace fast_deconv::morphology {

// Scatter form of the CUDA gather: out(r, c) is set when data(r - h + i, c - w + j) && structure(i, j).
void binary_dilation(const core::exec_ctx& ctx, core::span2d<bool> data, core::span2d<bool> structure,
                     common::roi structure_roi, core::span2d<bool> out)
{
  FD_PROFILE_FN();
  const int nrow = data.extent(0);
  const int ncol = data.extent(1);
  const int nrow_se = structure_roi.rmax - structure_roi.rmin;
  const int ncol_se = structure_roi.cmax - structure_roi.cmin;
  const int half_row = nrow_se / 2;
  const int half_col = ncol_se / 2;

  for (int r = 0; r < nrow; r++) {
    for (int c = 0; c < ncol; c++) out(r, c) = false;
  }

  for (int pr = 0; pr < nrow; pr++) {
    for (int pc = 0; pc < ncol; pc++) {
      if (!data(pr, pc)) continue;
      out(pr, pc) = true;
      for (int i = 0; i < nrow_se; i++) {
        const int r = pr + half_row - i;
        if (r < 0 || r >= nrow) continue;
        for (int j = 0; j < ncol_se; j++) {
          const int c = pc + half_col - j;
          if (c < 0 || c >= ncol) continue;
          if (structure(structure_roi.rmin + i, structure_roi.cmin + j)) out(r, c) = true;
        }
      }
    }
  }
}

}  // namespace fast_deconv::morphology
