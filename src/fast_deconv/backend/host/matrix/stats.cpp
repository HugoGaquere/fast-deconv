#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fast_deconv/core/profiler.hpp>
#include <fast_deconv/matrix/stats.hpp>
#include <limits>

namespace fast_deconv::matrix {

// TODO: use peak_criterion from peak.hpp

namespace {
// Masked pixels are already -inf in the data, so the max sentinel has to tie with
// them, not beat them: an all-masked image must report -inf, not a finite floor.
constexpr float kNoMax = -std::numeric_limits<float>::infinity();
}  // namespace

stats_impl::stats_impl(const core::exec_ctx& /*ctx*/, std::int64_t n_elements, bool use_abs)
    : n_elements_(n_elements), use_abs_(use_abs)
{
}

stats_acc stats_impl::accumulate(core::span2d<const float> data, core::span2d<const bool> mask) const
{
  FD_PROFILE_FN();
  const float* d = data.data_handle();
  const bool* m = mask.data_handle();

  // The mask excludes pixels from the max only; every pixel feeds the standard deviation, so
  // the noise estimate covers the whole image. Sums accumulate in double: a
  // sequential float sum drifts where the device tree reduction does not.
  float max_v = kNoMax;
  double sum = 0.0;
  double sum_sq = 0.0;
  const bool use_abs = use_abs_;
  const std::int64_t n = n_elements_;
#pragma omp parallel for reduction(max : max_v) reduction(+ : sum, sum_sq)
  for (std::int64_t i = 0; i < n; i++) {
    const float v = d[i];
    max_v = std::max(max_v, m[i] ? kNoMax : (use_abs ? std::fabs(v) : v));
    sum += v;
    sum_sq += static_cast<double>(v) * v;
  }

  return {max_v, static_cast<float>(sum), static_cast<float>(sum_sq), n};
}

}  // namespace fast_deconv::matrix
