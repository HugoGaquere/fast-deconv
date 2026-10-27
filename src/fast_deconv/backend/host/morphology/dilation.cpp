#include <cstdint>
#include <fast_deconv/core/profiler.hpp>
#include <fast_deconv/morphology/dilation.hpp>

namespace fast_deconv::morphology {

// Scatter form of the CUDA gather: out(r, c) is set when data(r - h + i, c - w + j) && structure(i, j).
void binary_dilation(const core::exec_ctx& ctx, core::span2d<const bool> data, core::span2d<const bool> structure,
                     core::roi structure_roi, core::span2d<bool> out)
{
  FD_PROFILE_FN();
  const std::int64_t nrow = data.extent(0);
  const std::int64_t ncol = data.extent(1);
  const std::int64_t nrow_se = structure_roi.rmax - structure_roi.rmin;
  const std::int64_t ncol_se = structure_roi.cmax - structure_roi.cmin;
  const std::int64_t half_row = nrow_se / 2;
  const std::int64_t half_col = ncol_se / 2;

  ctx.zero(out);

  for (std::int64_t pr = 0; pr < nrow; pr++) {
    for (std::int64_t pc = 0; pc < ncol; pc++) {
      if (!data(pr, pc)) continue;
      out(pr, pc) = true;
      for (std::int64_t i = 0; i < nrow_se; i++) {
        const std::int64_t r = pr + half_row - i;
        if (r < 0 || r >= nrow) continue;
        for (std::int64_t j = 0; j < ncol_se; j++) {
          const std::int64_t c = pc + half_col - j;
          if (c < 0 || c >= ncol) continue;
          if (structure(structure_roi.rmin + i, structure_roi.cmin + j)) out(r, c) = true;
        }
      }
    }
  }
}

}  // namespace fast_deconv::morphology
