#pragma once

#include <fast_deconv/common/region.hpp>
#include <fast_deconv/core/exec_ctx.hpp>
#include <vector>

namespace fast_deconv::common {

/**
 * @brief Build a per-scale mask by dilating each scale's peak set by the
 *        FWHM support of the central facet's double-convolved PSF (psf ** g_s ** g_s).
 *
 * Mirrors the conv2 half of `psf_convolution::build`, restricted to a single
 * facet: the per-frequency PSFs are weighted-averaged across channels into one
 * plane, transformed once, and per scale convolved with G(sigma_s * sqrt(2))
 * (= G_s twice) to obtain the 2D conv2_psf used for the FWHM mask.
 *
 * Once, for every scale:
 *   1. Zero mask_per_scale, then set mask_per_scale[scales[i], row, col] = true at every coords[i]
 *   2. Weighted channel mean of the PSFs, transformed once
 *
 * Then, for each scale s:
 *   3. Build conv2_psf[s] = G(sigma_s * sqrt(2)) ** (sum_c w[c] psf[c])
 *   4. FWHM bool mask: conv2_psf[s] > 0.5 * max(conv2_psf[s])
 *   5. Dilate mask_per_scale[s] using the FWHM mask as the structuring element
 *   6. Negate (so component-neighborhoods are valid=false), OR external_mask in
 *      (so externally-masked pixels stay masked everywhere).
 *
 * Output convention: true = masked, false = valid.
 *
 * @param[in]    ctx                 Execution context (work runs on it; also used for scratch allocations).
 * @param[in]    coords              Per-component peak coordinates (row, col).
 * @param[in]    scales              Per-component scale index, same length as coords.
 * @param[in]    central_facet_psfs  Central facet per-frequency PSFs, device, (n_freq, psf_h, psf_w).
 * @param[in]    weights_freq        Per-channel weights, device, (n_freq,).
 * @param[in]    scale_sigmas        Gaussian sigma per scale, host, (n_scales,).
 * @param[in]    external_mask       External mask, device, (dirty_h, dirty_w), true = masked; OR'd into every slice.
 * @param[out]   mask_per_scale      (n_scales, dirty_h, dirty_w) bool, written entirely.
 */
void build_auto_mask(const core::exec_ctx& ctx, const std::vector<index2d>& coords, const std::vector<int>& scales,
                     core::span3d<const float> central_facet_psfs, core::span1d<const float> weights_freq,
                     const std::vector<float>& scale_sigmas, core::span2d<const bool> external_mask,
                     core::span3d<bool> mask_per_scale);

/**
 * @brief Union of the valid regions of some scales' masks: the pixels at least one of them may clean.
 *
 * out[r, c] = false (valid) iff mask_per_scale[s, r, c] is false for some s in @p scales; with no scales,
 * every pixel is masked.
 *
 * @param[in]  ctx             Execution context.
 * @param[in]  mask_per_scale  (n_scales, nrow, ncol) bool, true = masked, as written by build_auto_mask.
 * @param[in]  scales          Indices of the scales to combine, host.
 * @param[out] out             (nrow, ncol) bool, true = masked, written entirely.
 */
void combine_scale_masks(const core::exec_ctx& ctx, core::span3d<const bool> mask_per_scale,
                         const std::vector<int>& scales, core::span2d<bool> out);

}  // namespace fast_deconv::common
