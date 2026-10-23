#include <algorithm>
#include <cassert>
#include <cmath>
#include <fast_deconv/matrix/stats.hpp>
#include <limits>

namespace fast_deconv::matrix {

stats_result stats_ctx::max_and_rms(core::span2d<const float> data, core::span2d<const bool> mask)
{
  assert(data.is_exhaustive() && mask.is_exhaustive());
  assert(data.extents() == mask.extents());
  assert(data.mapping().required_span_size() == n_elements_);

  const stats_acc acc = impl_.accumulate(data, mask);

  if (acc.count == 0) return {-std::numeric_limits<float>::infinity(), 0.f};

  const float inv_n = 1.f / static_cast<float>(acc.count);
  const float mean = acc.sum * inv_n;
  const float msq = acc.sum_sq * inv_n;
  return {acc.max_v, std::sqrt(std::max(msq - mean * mean, 0.f))};
}

}  // namespace fast_deconv::matrix
