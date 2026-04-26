#pragma once
#include <fast_deconv/core/span_types.hpp>
#include <fast_deconv/matrix/argmax.hpp>

namespace fast_deconv::common {

void subtract_component_async(const core::stream_resources& stream_res,
                              core::device_span2d<float> residual, core::device_span2d<float> psf,
                              std::pair<int, int> peak_coords, float gain);

void subtract_component_async(const core::stream_resources& stream_res,
                              core::device_span3d<float> residual, core::device_span3d<float> psf,
                              core::device_vect<float> spectral_coeffs,
                              std::pair<int, int> peak_coords, float gain);

}  // namespace fast_deconv::common