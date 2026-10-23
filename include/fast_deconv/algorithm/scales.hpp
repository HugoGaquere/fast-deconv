#pragma once
#include <fast_deconv/core/exec_ctx.hpp>
#include <fast_deconv/core/memory_types.hpp>
#include <fast_deconv/linalg/fft.hpp>
#include <fast_deconv/linalg/gaussian_convolution.hpp>
#include <fast_deconv/matrix/argmax.hpp>
#include <limits>
#include <vector>

namespace fast_deconv::scale {

/// Selected scale, its residual plane and that plane's peak.
struct scale_result {
  int scale;                            ///< index of the selected scale
  core::cont2d<float> scaled_residual;  ///< residual convolved with the selected scale
  matrix::peak peak;                    ///< peak value and flat index within scaled_residual
  matrix::peak_criterion criterion;     ///< how scaled_residual was ranked; mask points into the caller's mask
};

/**
 * @brief Pick the scale whose convolved residual has the highest biased peak.
 *
 * Each non-retired scale s is ranked by bias[s] * peak_s, where peak_s is the masked peak of @p dirty convolved
 * with Gaussian(sigmas[s]), ranked by matrix::peak_criterion. Scale 0 is taken as the identity, whatever
 * sigmas[0]: @p dirty is searched directly. Strictly greater wins, so the lowest scale takes a tie and a NaN peak
 * never wins. Syncs @p exec_ctx.
 *
 * @param[in] exec_ctx  Execution context; scratch is allocated on it.
 * @param[in] conv_ctx  Convolution context for one (nrow, ncol) image, batch 1, wrap-free for @p sigmas.
 * @param[in] dirty     Mean residual, device, (nrow, ncol).
 * @param[in] sigmas    Gaussian sigma per scale, (n_scales,).
 * @param[in] bias      Weight on each scale's peak, host, (n_scales,).
 * @param[in] retired   Indices of the scales to skip.
 * @param[in] mask      Device, true = masked: (n_scales, nrow, ncol) one plane per scale, (1, nrow, ncol) shared by
 *                      every scale, or a null data handle for no mask. Must outlive the returned criterion.
 * @param[in] absolute  Rank by |x| instead of x.
 * @return The selected scale, its convolved residual, that plane's peak and the criterion it was ranked by.
 * @throws std::invalid_argument If every scale is retired.
 */
scale_result select_best_scale(const core::exec_ctx& exec_ctx, const linalg::gaussian_convolution_ctx& conv_ctx,
                               core::span2d<const float> dirty, const std::vector<float>& sigmas,
                               core::host_span1d<const float> bias, const std::vector<int>& retired,
                               core::span3d<const bool> mask, bool absolute);

/// select_best_scale() with one (nrow, ncol) @p mask shared by every scale.
inline scale_result select_best_scale(const core::exec_ctx& exec_ctx, const linalg::gaussian_convolution_ctx& conv_ctx,
                                      core::span2d<const float> dirty, const std::vector<float>& sigmas,
                                      core::host_span1d<const float> bias, const std::vector<int>& retired,
                                      core::span2d<const bool> mask, bool absolute)
{
  return select_best_scale(exec_ctx, conv_ctx, dirty, sigmas, bias, retired,
                           core::span3d<const bool>(mask.data_handle(), 1, mask.extent(0), mask.extent(1)), absolute);
}

}  // namespace fast_deconv::scale
