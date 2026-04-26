#pragma once

#include <fast_deconv/core/resources.hpp>
#include <fast_deconv/core/span_types.hpp>

/**
 * @brief   Compute per-facet gains from single-convolved PSFs.
 * @details For each facet, computes the weighted mean of the single-convolved
 *          PSF over channels, then takes the max value. The gain is
 *          gamma / max(weighted_mean).
 *
 *          For scale 0 (delta), all gains are set to gamma directly.
 *
 * @param[in]  resources  GPU memory allocator.
 * @param[in]  stream_res CUDA stream resources.
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

std::vector<float> compute_gain_batched(const core::resources& resources,
                                        const core::stream_resources& stream_res,
                                        const core::device_span4d<float>& psfs,
                                        const core::device_vect<float>& weights_freq, float gamma);
}  // namespace fast_deconv::gain
