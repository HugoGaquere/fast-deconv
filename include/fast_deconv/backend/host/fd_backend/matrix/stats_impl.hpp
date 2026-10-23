#pragma once

#include <cstdint>
#include <fast_deconv/core/exec_ctx.hpp>
#include <fast_deconv/matrix/stats_acc.hpp>

namespace fast_deconv::matrix {

/// Host backend of stats_ctx
class stats_impl {
 public:
  stats_impl(const core::exec_ctx& ctx, std::int64_t n_elements, bool use_abs);

  /// The fused reduction over @p data, the max skipping pixels where @p mask is true.
  stats_acc accumulate(core::span2d<const float> data, core::span2d<const bool> mask) const;

 private:
  std::int64_t n_elements_;
  bool use_abs_;
};

}  // namespace fast_deconv::matrix
