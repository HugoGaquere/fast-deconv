#include <algorithm>
#include <cassert>
#include <cstdint>
#include <fast_deconv/core/macro.hpp>
#include <fast_deconv/core/profiler.hpp>
#include <fast_deconv/morphology/roi.hpp>
#include <fd_backend/core/reduce.hpp>

namespace fast_deconv::morphology {

namespace {

// Identity of roi_union: an inverted box that any set pixel replaces.
constexpr core::roi kNoRoi{INT64_MAX, 0, INT64_MAX, 0};

// Pixel i as a one-pixel box if set, the identity otherwise.
struct pixel_roi {
  const bool* data;
  std::int64_t ncol;

  FD_HOST_DEVICE core::roi operator()(std::int64_t i) const
  {
    if (!data[i]) return kNoRoi;
    const std::int64_t r = i / ncol;
    const std::int64_t c = i % ncol;
    return {r, r + 1, c, c + 1};
  }
};

// Bounding box of two boxes.
struct roi_union {
  FD_HOST_DEVICE core::roi operator()(const core::roi& a, const core::roi& b) const
  {
    return {std::min(a.rmin, b.rmin), std::max(a.rmax, b.rmax), std::min(a.cmin, b.cmin), std::max(a.cmax, b.cmax)};
  }
};

}  // namespace

core::roi compute_mask_roi(const core::exec_ctx& ctx, core::span2d<const bool> data)
{
  FD_PROFILE_FN();
  assert(data.is_exhaustive());
  const core::roi result =
      core::reduce(ctx, core::count(data), pixel_roi{data.data_handle(), data.extent(1)}, roi_union{}, kNoRoi);

  // Nothing raised rmax, so the mask was empty: report an empty box, not the identity.
  if (result.rmax == 0) return {};
  return result;
}

}  // namespace fast_deconv::morphology
