#pragma once

#include <cstddef>
#include <cstdint>
#include <fast_deconv/core/exec_ctx.hpp>

namespace fast_deconv::matrix {

/**
 * @brief Packed reduction state combining the four contributions needed to compute the max and the standard
 *        deviation in a single sweep.
 *
 * Public because run_async() leaves it on backend memory for the caller to finalize.
 */
struct stats_acc {
  float max_v;
  float sum;
  float sum_sq;
  std::int64_t count;
};

/// Result returned to the host after a fused max + standard deviation reduction.
struct stats_result {
  float max;  ///< masked max (with optional abs); -inf if every pixel is masked
  /// Standard deviation, not the root mean square: sqrt(max(E[x^2] - E[x]^2, 0)), over all pixels (mask not
  /// applied). Named rms after the noise estimate it stands for; equal to the RMS only for a zero-mean image.
  float rms;
};

/**
 * @brief Reusable state for the fused max + standard deviation reduction over a 2D image with a boolean mask.
 *
 * On the CUDA backend the constructor runs the CUB temp-storage sizing query once, so the minor-cycle loop only
 * pays for the reduction itself; `use_abs` is fixed at construction so the iterator type, and with it that
 * query, stays valid for every subsequent call. The host backend has no scratch to size.
 */
class stats_ctx {
 public:
  /**
   * @param[in] ctx         Context the reductions run on. Must outlive this.
   * @param[in] n_elements  Pixels per image.
   * @param[in] use_abs     Take the max of |x| instead of x.
   */
  stats_ctx(const core::exec_ctx& ctx, std::int64_t n_elements, bool use_abs);

  stats_ctx(const stats_ctx&) = delete;
  stats_ctx& operator=(const stats_ctx&) = delete;

  /**
   * @brief Issue the fused reduction.
   *
   * No host sync; the result stays in backend memory, reachable through device_state().
   *
   * @param[in] data  Image, n_elements pixels.
   * @param[in] mask  Same shape as @p data, true = excluded from the max (the standard deviation ignores it).
   */
  void run_async(core::span2d<const float> data, core::span2d<const bool> mask);

  /// Async issue + D2H copy + ctx sync. Returns {max, standard deviation} on host.
  stats_result run(core::span2d<const float> data, core::span2d<const bool> mask);

  /// The packed accumulator left by run_async(), in backend memory.
  const stats_acc* device_state() const { return d_state_.get(); }

 private:
  const core::exec_ctx& ctx_;
  std::int64_t n_elements_;
  bool use_abs_;
  std::size_t temp_bytes_ = 0;
  core::owned_ptr<std::byte> d_temp_;
  core::owned_ptr<stats_acc> d_state_;
  stats_acc h_state_{};
};

}  // namespace fast_deconv::matrix
