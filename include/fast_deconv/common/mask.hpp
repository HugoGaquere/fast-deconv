#pragma once

#include <fast_deconv/common/region.hpp>
#include <fast_deconv/core/exec_ctx.hpp>
#include <vector>

namespace fast_deconv::common {

void mask_and_abs_async(const core::exec_ctx& ctx, core::span2d<float> data, core::span2d<bool> mask, float fill_value,
                        bool abs);

void mask_and_abs_async(const core::exec_ctx& ctx, core::span3d<float> data, core::span2d<bool> mask, float fill_value,
                        bool abs);

void mask_and_abs_async(const core::exec_ctx& ctx, core::span3d<float> data, core::span3d<bool> mask, float fill_value,
                        bool abs);

void mask_less_than_threshold(const core::exec_ctx& ctx, core::span2d<float> data, float threshold, float fill_value);

/**
 * @brief Build a per-scale mask by dilating each scale's peak set by the
 *        FWHM support of the central facet's double-convolved PSF (psf ** g_s ** g_s).
 *
 * Mirrors the `convolve_psfs_with_scale_async` pipeline, restricted to a single
 * facet and one scale at a time and producing only the doubly-convolved PSF:
 * the per-frequency PSFs are FFT'd in batch once, then per scale the freq
 * arrays are multiplied by G_s^2, IFFT'd batch-back to space, and finally
 * weighted-averaged across channels to obtain the 2D conv2_psf used for the
 * FWHM mask.
 *
 * For each scale s:
 *   1. Premask: set mask_per_scale[s, row, col] = true at every (coords[i], col) where scales[i] == s
 *   2. Build conv2_psf[s] = mean_c w[c] * ifft( fft(psf[c]) * G_s^2 )
 *   3. FWHM bool mask: conv2_psf[s] > 0.5 * max(conv2_psf[s])
 *   4. Dilate mask_per_scale[s] using the FWHM mask as the structuring element
 *   5. Negate (so component-neighborhoods are valid=false), OR external_mask in
 *      (so externally-masked pixels stay masked everywhere).
 *
 * Output convention matches `mask_and_abs_async`: true = masked (filled), false = valid.
 *
 * @param[in]    ctx                 Execution lane (kernels run on its stream; also used for scratch allocations).
 * @param[in]    coords              Per-component peak coordinates (row, col).
 * @param[in]    scales              Per-component scale index, same length as coords.
 * @param[in]    central_facet_psfs  Central facet per-frequency PSFs, (n_freq, psf_h, psf_w).
 * @param[in]    weights_freq        Per-channel weights, device, (n_freq,).
 * @param[in]    scale_sigmas        Gaussian sigma per scale, device, (n_scales,).
 * @param[in]    fft_padding         FFT padding factor used to size the conv plans.
 * @param[in]    external_mask       Externally-supplied 2D mask (true=masked) OR'd into every scale slice.
 * @param[out]   mask_per_scale      (n_scales, dirty_h, dirty_w) bool, written entirely.
 */
void build_auto_mask(const core::exec_ctx& ctx, const std::vector<index2d>& coords, const std::vector<int>& scales,
                     core::span3d<float> central_facet_psfs, core::span1d<const float> weights_freq,
                     core::span1d<float> scale_sigmas, float fft_padding, core::span2d<bool> external_mask,
                     core::span3d<bool> mask_per_scale);

}  // namespace fast_deconv::common
