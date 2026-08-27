#pragma once

#include <cstddef>
#include <cub/version.cuh>
#include <fast_deconv/core/exec_ctx.hpp>
#include <fast_deconv/core/memory_types.hpp>
#include <fast_deconv/matrix/peak.hpp>

#if CUB_VERSION < 300000
#include <cub/util_type.cuh>  // cub::KeyValuePair
#endif

namespace fast_deconv::matrix {

/// Reusable state for a CUB argmax over a fixed element count on one lane.
/// The constructor runs the CUB temp-storage sizing query once so the
/// minor-cycle loop only pays for the reduction itself.
class argmax_ctx {
 public:
  argmax_ctx(const core::exec_ctx& ctx, std::size_t n_elements);

  argmax_ctx(const argmax_ctx&) = delete;
  argmax_ctx& operator=(const argmax_ctx&) = delete;

  /// Issue the reduction; the result stays on device until argmax() reads it.
  void run_async(core::span2d<float> data);

  /// Async issue + D2H copy + stream sync.
  peak run(core::span2d<float> data);

  std::size_t size() const { return n_elements_; }

 private:
  const core::exec_ctx& ctx_;
  std::size_t n_elements_;
  std::size_t temp_bytes_ = 0;
  core::owned_ptr<std::byte> d_temp_;
#if CUB_VERSION >= 300000
  core::owned_ptr<float> d_peak_value_;
  core::owned_ptr<std::int64_t> d_peak_index_;
  float h_peak_value_ = 0.0f;
  std::int64_t h_peak_index_ = 0;
#else
  core::owned_ptr<cub::KeyValuePair<int, float> > d_peak_;
  cub::KeyValuePair<int, float> h_peak_{};
#endif
};

}  // namespace fast_deconv::matrix
