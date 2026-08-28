#pragma once

#include <fast_deconv/core/exec_ctx.hpp>
#include <vector>

/**
 * @brief   Compute per-facet gains from single-convolved PSFs.
 * @details For each facet, computes the weighted mean of the single-convolved
 *          PSF over channels, then takes the max value. The gain is
 *          gamma / max(weighted_mean).
 *
 *          For scale 0 (delta), all gains are set to gamma directly.
 *
 * @param[in]  ctx        Execution lane (also used for scratch allocations).
 * @param[in]  conv_psfs  Single-convolved PSFs, device,
 *                        layout (n_facets, nch, psf_npix), pre-computed by
 *                        convolve_psfs_for_scale.
 * @param[in]  weights    Per-channel weights, device, size nch.
 * @param[in]  n_facets   Number of facets.
 * @param[in]  nch        Number of frequency channels.
 * @param[in]  psf_npix   Number of spatial pixels per PSF (psf_nrow * psf_ncol).
 * @param[in]  scale_idx  Index of the selected scale (0 = delta).
 * @param[in]  gamma      CLEAN loop gain parameter.
 *
 * @return Per-facet gains, host vector of size n_facets.
 */

namespace fast_deconv::common {

std::vector<float> compute_gain_batched(const core::exec_ctx& ctx, const core::span4d<float>& psfs,
                                        const core::span1d<const float>& weights_freq, float gamma);

std::vector<float> compute_all_gains_batched(const core::exec_ctx& ctx, const core::span5d<float>& psfs,
                                             const core::span1d<const float>& weights_freq, float gamma);
}  // namespace fast_deconv::common
