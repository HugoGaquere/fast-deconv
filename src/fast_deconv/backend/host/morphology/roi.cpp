#include <algorithm>
#include <climits>
#include <fast_deconv/morphology/roi.hpp>

namespace fast_deconv::morphology {

common::roi compute_mask_roi(const core::exec_ctx& ctx, core::span2d<bool> data)
{
  common::roi result{INT_MAX, 0, INT_MAX, 0};

  for (int r = 0; r < data.extent(0); r++) {
    for (int c = 0; c < data.extent(1); c++) {
      if (!data(r, c)) continue;
      result.rmin = std::min(result.rmin, r);
      result.rmax = std::max(result.rmax, r + 1);
      result.cmin = std::min(result.cmin, c);
      result.cmax = std::max(result.cmax, c + 1);
    }
  }

  // Nothing raised rmax, so the mask was empty: report an empty box, not the identity.
  if (result.rmax == 0) return {};
  return result;
}

}  // namespace fast_deconv::morphology
