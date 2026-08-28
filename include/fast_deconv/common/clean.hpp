#pragma once
#include <fast_deconv/common/region.hpp>
#include <fast_deconv/core/exec_ctx.hpp>

namespace fast_deconv::common {

void subtract_component_async(const core::exec_ctx& ctx, core::span2d<float> residual, core::span2d<float> psf,
                              index2d peak_coords, float gain);

void subtract_component_async(const core::exec_ctx& ctx, core::span3d<float> residual, core::span3d<float> psf,
                              core::span1d<float> spectral_coeffs, index2d peak_coords, float gain);

}  // namespace fast_deconv::common