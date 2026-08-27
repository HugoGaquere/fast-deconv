#pragma once

#include <cstddef>
#include <fast_deconv/core/exec_ctx.hpp>

namespace fast_deconv::matrix {

/// Packed reduction state combining the four contributions needed to compute
/// max + rms in a single sweep. Public because run_async() leaves it on
/// backend memory for the caller to finalize.
struct stats_acc {
  float max_v;
  float sum;
  float sum_sq;
  int count;
};

/// Result returned to the host after a fused max + rms reduction.
struct stats_result {
  float max;  ///< masked max (with optional abs); -FLT_MAX if every pixel is masked
  float rms;  ///< sqrt(max(E[x^2] - E[x]^2, 0)) over all pixels (mask not applied)
};

/// Reusable state for the fused max + rms reduction over a 2D image with a
/// boolean mask. The constructor runs the temp-storage sizing query once so
/// the minor-cycle loop only pays for the reduction itself.
///
/// `use_abs` is fixed at construction so the iterator type stays stable and
/// the temp-bytes query computed in the constructor stays valid for every
/// subsequent call.
class stats_ctx {
 public:
  stats_ctx(const core::exec_ctx& ctx, std::size_t n_elements, bool use_abs);

  stats_ctx(const stats_ctx&) = delete;
  stats_ctx& operator=(const stats_ctx&) = delete;

  /// Issue the fused reduction. No host sync; the result stays in backend
  /// memory, reachable through device_state().
  void run_async(core::span2d<float> data, core::span2d<bool> mask);

  /// Async issue + D2H copy + lane sync. Returns {max, rms} on host.
  stats_result run(core::span2d<float> data, core::span2d<bool> mask);

  /// The packed accumulator left by run_async(), in backend memory.
  const stats_acc* device_state() const { return d_state_.get(); }

  std::size_t size() const { return n_elements_; }

 private:
  const core::exec_ctx& ctx_;
  std::size_t n_elements_;
  bool use_abs_;
  std::size_t temp_bytes_ = 0;
  core::owned_ptr<std::byte> d_temp_;
  core::owned_ptr<stats_acc> d_state_;
  stats_acc h_state_{};
};

}  // namespace fast_deconv::matrix
