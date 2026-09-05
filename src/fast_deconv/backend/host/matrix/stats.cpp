#include <algorithm>
#include <cassert>
#include <cmath>
#include <fast_deconv/matrix/stats.hpp>
#include <limits>

namespace fast_deconv::matrix {

namespace {
// Masked pixels are already -inf in the data, so the max sentinel has to tie with
// them, not beat them: an all-masked image must report -inf, not a finite floor.
constexpr float kNoMax = -std::numeric_limits<float>::infinity();
}  // namespace

stats_ctx::stats_ctx(const core::exec_ctx& ctx, std::size_t n_elements, bool use_abs)
    : ctx_(ctx), n_elements_(n_elements), use_abs_(use_abs)
{
  // No reduction scratch to size, but device_state() must stay reachable.
  d_state_ = ctx_.alloc_ptr_async<stats_acc>(1);
}

void stats_ctx::run_async(core::span2d<float> data, core::span2d<bool> mask)
{
  assert(data.is_exhaustive() && mask.is_exhaustive());
  assert(data.extents() == mask.extents());
  assert(data.size() == n_elements_);

  const float* d = data.data_handle();
  const bool* m = mask.data_handle();

  // The mask excludes pixels from the max only; every pixel feeds the RMS, so
  // the noise estimate covers the whole image. Sums accumulate in double: a
  // sequential float sum drifts where the device tree reduction does not.
  float max_v = kNoMax;
  double sum = 0.0;
  double sum_sq = 0.0;
  for (std::size_t i = 0; i < n_elements_; i++) {
    const float v = d[i];
    max_v = std::max(max_v, m[i] ? kNoMax : (use_abs_ ? std::fabs(v) : v));
    sum += v;
    sum_sq += static_cast<double>(v) * v;
  }

  d_state_[0] = {max_v, static_cast<float>(sum), static_cast<float>(sum_sq), static_cast<int>(n_elements_)};
}

stats_result stats_ctx::run(core::span2d<float> data, core::span2d<bool> mask)
{
  run_async(data, mask);
  h_state_ = d_state_[0];

  if (h_state_.count == 0) return {kNoMax, 0.f};

  const float inv_n = 1.f / static_cast<float>(h_state_.count);
  const float mean = h_state_.sum * inv_n;
  const float msq = h_state_.sum_sq * inv_n;
  return {h_state_.max_v, std::sqrt(std::max(msq - mean * mean, 0.f))};
}

}  // namespace fast_deconv::matrix
