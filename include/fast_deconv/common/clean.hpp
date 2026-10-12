#pragma once
#include <fast_deconv/common/region.hpp>
#include <cmath>
#include <fast_deconv/core/exec_ctx.hpp>
#include <fast_deconv/matrix/peak.hpp>

namespace fast_deconv::common {

/// Pixels where criterion(residual) < @p threshold are left untouched; the defaults subtract everywhere.
void subtract_component_async(const core::exec_ctx& ctx, core::span2d<float> residual, core::span2d<const float> psf,
                              index2d peak_coords, float gain, matrix::peak_criterion criterion = {},
                              float threshold = -INFINITY);

void subtract_component_async(const core::exec_ctx& ctx, core::span3d<float> residual, core::span3d<const float> psf,
                              core::span1d<const float> spectral_coeffs, index2d peak_coords, float gain);

}  // namespace fast_deconv::common
