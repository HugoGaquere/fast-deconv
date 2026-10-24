#pragma once

#include <cstdint>
#include <fast_deconv/core/exec_ctx.hpp>

namespace fast_deconv::matrix {

/// Result returned to the host after a fused max + standard deviation reduction.
struct stats_result {
  float max;  ///< masked max (with optional abs); -inf if every pixel is masked
  float rms;  ///< Standard deviation: sqrt(max(E[x^2] - E[x]^2, 0)), over all pixels (mask not applied).
};

/**
 * @brief Max and standard deviation of @p data, on host, in one reduction. Syncs @p ctx.
 *
 * The sums accumulate in the backend's stats_sum_type.
 *
 * @param[in] ctx      Execution context.
 * @param[in] data     Image.
 * @param[in] mask     Same shape as @p data, true = excluded from the max (the standard deviation ignores it).
 * @param[in] use_abs  Take the max of |x| instead of x.
 */
stats_result max_and_rms(const core::exec_ctx& ctx, core::span2d<const float> data, core::span2d<const bool> mask,
                         bool use_abs);

}  // namespace fast_deconv::matrix
