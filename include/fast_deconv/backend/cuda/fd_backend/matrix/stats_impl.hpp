#pragma once

#include <cstddef>
#include <cstdint>
#include <fast_deconv/core/exec_ctx.hpp>
#include <fast_deconv/matrix/stats_acc.hpp>

namespace fast_deconv::matrix {

/**
 * @brief CUDA backend of stats_ctx: a CUB reduction into a device accumulator, read back on host.
 *
 * The constructor runs the CUB temp-storage sizing query once, so the minor-cycle loop only pays for the
 * reduction itself; `use_abs` is fixed at construction so the iterator type, and with it that query, stays valid
 * for every subsequent call.
 */
class stats_impl {
 public:
  stats_impl(const core::exec_ctx& ctx, std::int64_t n_elements, bool use_abs);

  /// The fused reduction over @p data, the max skipping pixels where @p mask is true. Syncs @p ctx.
  stats_acc accumulate(core::span2d<const float> data, core::span2d<const bool> mask);

 private:
  const core::exec_ctx& ctx_;
  std::int64_t n_elements_;
  bool use_abs_;
  std::size_t temp_bytes_ = 0;
  core::owned_ptr<std::byte> d_temp_;
  core::owned_ptr<stats_acc> d_acc_;
};

}  // namespace fast_deconv::matrix
