#pragma once

#include <fast_deconv/common/region.hpp>
#include <fast_deconv/core/exec_ctx.hpp>

namespace fast_deconv::multi_frequency {

void fit_coefficients(const core::exec_ctx& ctx, const core::span3d<float> residual,
                      const core::span3d<const float> jones_norm, const core::span1d<const float> weights_freq,
                      const core::span2d<float> xdes, const common::index2d peak_coords,
                      core::span1d<float> compact_coeffs_out, core::span1d<float> coeffs_per_chan_out);

}