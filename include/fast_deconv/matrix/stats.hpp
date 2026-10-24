#pragma once

#include <concepts>
#include <cstdint>
#include <fast_deconv/core/exec_ctx.hpp>

// Resolved by the include path: CMake puts backend/${FAST_DECONV_BACKEND}
// on it, and every backend provides this file.
#include <fd_backend/matrix/stats_impl.hpp>

namespace fast_deconv::matrix {

/// Result returned to the host after a fused max + standard deviation reduction.
struct stats_result {
  float max;  ///< masked max (with optional abs); -inf if every pixel is masked
  float rms;  ///< Standard deviation: sqrt(max(E[x^2] - E[x]^2, 0)), over all pixels (mask not applied).
};

namespace detail {
/// What every backend's fd_backend/matrix/stats_impl.hpp must provide.
template <typename T>
concept stats_backend = std::constructible_from<T, const core::exec_ctx&, std::int64_t, bool> &&
                        requires(T impl, core::span2d<const float> data, core::span2d<const bool> mask) {
                          { impl.accumulate(data, mask) } -> std::same_as<stats_acc>;
                        };

static_assert(stats_backend<stats_impl>, "fd_backend/matrix/stats_impl.hpp must model stats_backend");
}  // namespace detail

/**
 * @brief Reusable state for the fused max + standard deviation reduction over a 2D image with a boolean mask.
 *
 * The backend's stats_impl produces the packed accumulator; this layer checks the inputs and finalizes it.
 */
class stats_ctx {
 public:
  /**
   * @param[in] ctx         Context the reductions run on. Must outlive this.
   * @param[in] n_elements  Pixels per image.
   * @param[in] use_abs     Take the max of |x| instead of x.
   */
  stats_ctx(const core::exec_ctx& ctx, std::int64_t n_elements, bool use_abs)
      : n_elements_(n_elements), impl_(ctx, n_elements, use_abs)
  {
  }

  stats_ctx(const stats_ctx&) = delete;
  stats_ctx& operator=(const stats_ctx&) = delete;

  /**
   * @brief Max and standard deviation of @p data, on host.
   *
   * @param[in] data  Image, n_elements pixels.
   * @param[in] mask  Same shape as @p data, true = excluded from the max (the standard deviation ignores it).
   */
  stats_result max_and_rms(core::span2d<const float> data, core::span2d<const bool> mask);

 private:
  std::int64_t n_elements_;
  stats_impl impl_;
};

}  // namespace fast_deconv::matrix
