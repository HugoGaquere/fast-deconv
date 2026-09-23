#pragma once

#include <fast_deconv/core/exec_ctx.hpp>
#include <vector>

namespace fast_deconv::common {

/**
 * @brief  CLEAN gain for one facet's single-convolved PSF: gamma / max(weighted mean over channels).
 *
 * @param[in] ctx           Execution lane; scratch is allocated on it and the call syncs it.
 * @param[in] psf           Single-convolved PSFs, device, (n_freq, psf_h, psf_w).
 * @param[in] weights_freq  Per-channel weights, device, (n_freq,).
 * @param[in] gamma         CLEAN loop gain.
 */
float compute_gain(const core::exec_ctx& ctx, const core::span3d<const float>& psf,
                   const core::span1d<const float>& weights_freq, float gamma);

/// compute_gain for each facet of @p psfs, (n_facets, n_freq, psf_h, psf_w); returns n_facets gains.
std::vector<float> compute_gain_batched(const core::exec_ctx& ctx, const core::span4d<const float>& psfs,
                                        const core::span1d<const float>& weights_freq, float gamma);
}  // namespace fast_deconv::common
