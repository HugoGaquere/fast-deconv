#pragma once
#include <fast_deconv/core/exec_ctx.hpp>
#include <fast_deconv/core/memory_types.hpp>
#include <fast_deconv/linalg/convolution.hpp>
#include <fast_deconv/linalg/fft.hpp>
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

scale_result select_best_scale(const core::exec_ctx& exec_ctx, const linalg::convolution_ctx& conv_ctx,
                               core::span2d<const float> dirty, const std::vector<float>& sigmas,
                               core::host_span1d<float> bias, const std::vector<int>& retired, core::span3d<bool> mask,
                               bool absolute);

inline scale_result select_best_scale(const core::exec_ctx& exec_ctx, const linalg::convolution_ctx& conv_ctx,
                                      core::span2d<const float> dirty, const std::vector<float>& sigmas,
                                      core::host_span1d<float> bias, const std::vector<int>& retired,
                                      core::span2d<bool> mask, bool absolute)
{
  return select_best_scale(exec_ctx, conv_ctx, dirty, sigmas, bias, retired,
                           core::span3d<bool>(mask.data_handle(), 1, mask.extent(0), mask.extent(1)), absolute);
}

}  // namespace fast_deconv::scale
