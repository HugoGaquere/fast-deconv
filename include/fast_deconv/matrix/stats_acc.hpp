#pragma once

#include <cstdint>

namespace fast_deconv::matrix {

/**
 * @brief Packed reduction state combining the four contributions needed to compute the max and the standard
 *        deviation in a single sweep.
 *
 * The contract between stats_ctx and each backend's stats_impl, which returns it on host.
 */
struct stats_acc {
  float max_v;
  float sum;
  float sum_sq;
  std::int64_t count;
};

}  // namespace fast_deconv::matrix
