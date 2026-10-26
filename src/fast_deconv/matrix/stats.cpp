#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <fast_deconv/core/macro.hpp>
#include <fast_deconv/core/profiler.hpp>
#include <fast_deconv/matrix/stats.hpp>
#include <fd_backend/core/reduce.hpp>
#include <fd_backend/matrix/stats_types.hpp>

namespace fast_deconv::matrix {

namespace {

// The max and the two sums the standard deviation needs, in one sweep.
struct stats_acc {
  float max_v;
  stats_sum_type sum;
  stats_sum_type sum_sq;
};

// -inf is the masked-pixel value, so the identity ties with it instead of beating it: an all-masked image
// reports -inf, not a finite floor.
constexpr stats_acc kNoStats{-INFINITY, 0, 0};

// Pixel i: excluded from the max where masked, but every pixel feeds the standard deviation, as DDFacet's
// np.std(MeanDirty) stop threshold does.
struct pixel_stats {
  const float* data;
  const bool* mask;
  bool use_abs;

  FD_HOST_DEVICE stats_acc operator()(std::int64_t i) const
  {
    const float v = data[i];
    const stats_sum_type s = v;
    return {mask[i] ? -INFINITY : (use_abs ? fabsf(v) : v), s, s * s};
  }
};

struct stats_combine {
  FD_HOST_DEVICE stats_acc operator()(const stats_acc& a, const stats_acc& b) const
  {
    return {fmaxf(a.max_v, b.max_v), a.sum + b.sum, a.sum_sq + b.sum_sq};
  }
};

}  // namespace

stats_result max_and_rms(const core::exec_ctx& ctx, core::span2d<const float> data, core::span2d<const bool> mask,
                         bool use_abs)
{
  FD_PROFILE_FN();
  assert(data.is_exhaustive() && mask.is_exhaustive());
  assert(data.extents() == mask.extents());

  const std::int64_t n = core::count(data);
  if (n == 0) return {-INFINITY, 0.f};

  const stats_acc acc =
      core::reduce(ctx, n, pixel_stats{data.data_handle(), mask.data_handle(), use_abs}, stats_combine{}, kNoStats);

  const float inv_n = 1.f / static_cast<float>(n);
  const float mean = static_cast<float>(acc.sum) * inv_n;
  const float msq = static_cast<float>(acc.sum_sq) * inv_n;
  return {acc.max_v, std::sqrt(std::max(msq - mean * mean, 0.f))};
}

}  // namespace fast_deconv::matrix
