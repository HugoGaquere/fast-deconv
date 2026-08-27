#pragma once

#include <fast_deconv/core/exec_ctx.hpp>
#include <utility>

namespace fast_deconv::multi_frequency {

void fit_coefficients(const core::exec_ctx& ctx, const core::span3d<float> residual,
                      const core::span3d<float> jones_norm, const core::span1d<float> weights_freq,
                      const core::span2d<float> xdes, const std::pair<int, int> peak_coords,
                      core::span1d<float> compact_coeffs_out, core::span1d<float> coeffs_per_chan_out);

}