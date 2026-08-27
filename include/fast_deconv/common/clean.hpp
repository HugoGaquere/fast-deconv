#pragma once
#include <fast_deconv/core/exec_ctx.hpp>
#include <utility>

namespace fast_deconv::common {

void subtract_component_async(const core::exec_ctx& ctx, core::span2d<float> residual, core::span2d<float> psf,
                              std::pair<int, int> peak_coords, float gain);

void subtract_component_async(const core::exec_ctx& ctx, core::span3d<float> residual, core::span3d<float> psf,
                              core::span1d<float> spectral_coeffs, std::pair<int, int> peak_coords, float gain);

}  // namespace fast_deconv::common