#pragma once

#include <fast_deconv/algorithm/ddmsc_types.hpp>
#include <fast_deconv/core/exec_ctx.hpp>
#include <fast_deconv/core/span_types.hpp>

namespace fast_deconv::algorithm::ddmsc {

/**
 * @brief   Run the full minor-cycle loop: repeated scale selection with stall/divergence checks.
 * @details Allocates mean_residual internally and computes it from the dirty image at each
 *          iteration. PSFs are convolved on-the-fly for each selected scale.
 *
 * @param[in,out] ctx          DDMSC context (resources, FFT plans, raw PSFs, mask, biases, etc.).
 * @param[in]     p            Algorithm parameters.
 * @param[in,out] dirty        Multi-frequency dirty image (n_freq, nrow, ncol).
 * @param[in]     jones_norm   Jones normalization.
 * @param[in]     weights_freq Per-frequency weights.
 *
 * @return ddmsc_result with all extracted components, final flux, and iteration count.
 */
ddmsc_result run_ddmsc_cycles(context& ctx, const params& p, core::span3d<float>& dirty,
                              const core::span3d<float>& jones_norm, const core::span1d<float>& weights_freq);

}  // namespace fast_deconv::algorithm::ddmsc
