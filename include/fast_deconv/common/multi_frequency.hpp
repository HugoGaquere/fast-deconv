#pragma once

#include <fast_deconv/core/resources.hpp>
#include <fast_deconv/core/span_types.hpp>

namespace fast_deconv::multi_frequency {

void fit_coefficients(const core::resources& resources, const core::stream_resources& stream_res,
                      const core::device_span3d<float> residual,
                      const core::device_span3d<float> jones_norm,
                      const core::device_vect<float> weights_freq,
                      const core::device_span2d<float> xdes, const std::pair<int, int> peak_coords,
                      core::device_vect<float> compact_coeffs_out,
                      core::device_vect<float> coeffs_per_chan_out);

}