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
 * For each scale s:
 *   1. Premask: set mask_per_scale[s, row, col] = true at every coords[i] where scales[i] == s
 *   2. Build conv2_psf[s] = G(sigma_s * sqrt(2)) ** (sum_c w[c] psf[c])
 *   3. FWHM bool mask: conv2_psf[s] > 0.5 * max(conv2_psf[s])
 *   4. Dilate mask_per_scale[s] using the FWHM mask as the structuring element
 *   5. Negate (so component-neighborhoods are valid=false), OR external_mask in
 *      (so externally-masked pixels stay masked everywhere).
 *
 * Output convention: true = masked, false = valid.
 *
 * @param[in]    ctx                 Execution lane (kernels run on its stream; also used for scratch allocations).
 * @param[in]    coords              Per-component peak coordinates (row, col).
 * @param[in]    scales              Per-component scale index, same length as coords.
 * @param[in]    central_facet_psfs  Central facet per-frequency PSFs, (n_freq, psf_h, psf_w).
 * @param[in]    weights_freq        Per-channel weights, device, (n_freq,).
 * @param[in]    scale_sigmas        Gaussian sigma per scale, host, (n_scales,).
 * @param[in]    external_mask       Externally-supplied 2D mask (true=masked) OR'd into every scale slice.
 * @param[out]   mask_per_scale      (n_scales, dirty_h, dirty_w) bool, written entirely.
 */
void build_auto_mask(const core::exec_ctx& ctx, const std::vector<index2d>& coords, const std::vector<int>& scales,
                     core::span3d<float> central_facet_psfs, core::span1d<const float> weights_freq,
                     const std::vector<float>& scale_sigmas, core::span2d<bool> external_mask,
                     core::span3d<bool> mask_per_scale);

}  // namespace fast_deconv::common
