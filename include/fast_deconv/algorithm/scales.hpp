#pragma once
#include <fast_deconv/core/exec_ctx.hpp>
#include <fast_deconv/core/span_types.hpp>
#include <fast_deconv/linalg/fft.hpp>
#include <vector>

namespace fast_deconv::scale {

/**
 * @brief   Build Gaussian scale kernels in half-complex frequency space.
 * @details Each thread computes one (row, col) frequency bin for all scales.
 *
 * @param[in]  sigmas          Gaussian sigmas, device, shape (n_scales,).
 * @param[in]  scale_ncol_full Full spatial column count (used for frequency normalization).
 * @param[out] scales          Output kernels, device,
 *                             shape (n_scales, scale_nrow, scale_ncol_half).
 */
void make_gaussian_kernels_async(const core::exec_ctx& stream_res, core::span1d<float> sigmas, int scale_ncol_full,
                                 core::span3d<float> scales);

/**
 * @brief   Convolve a 2D mean residual image with Gaussian scale kernels.
 * @details Internally: pad+ifftshift -> R2C (once) -> per-scale: multiply -> C2R -> fftshift+crop.
 *
 * @param[in]  conv             Convolution plans over the dirty grid: forward_batch 1,
 *                              backward_batch dividing (n_scales - 1), one backward plan.
 * @param[in]  dirty            Mean residual, device, shape (nrow, ncol).
 * @param[in]  scales           Gaussian kernels in freq domain, device,
 *                              shape (n_scales, freq_nrow, freq_ncol).
 * @param[out] out_scaled_dirty Per-scale convolved output, device,
 *                              shape (n_scales, nrow, ncol).
 */
void convolve_with_scales(const linalg::convolve_ctx& conv, core::span2d<float> dirty, core::span3d<float> scales,
                          core::span3d<float> out_scaled_dirty);

/**
 * @brief   Finds the best scale and peak pixel via biased peak-finding.
 *
 * @param[in,out] scaled_dirty   Per-scale residuals, device, shape (n_scales, nrow, ncol).
 * @param[in]     bias           Per-scale bias, host, shape (n_scales,).
 * @param[in]     retired_scales Scale indices to exclude from selection.
 * @return Unbiased peak value and pixel coordinates of the selected scale.
 */
int scale_selection(const core::exec_ctx& stream_res, core::span3d<float> scaled_dirty, core::host_vect<float> bias,
                    const std::vector<int>& retired_scales);

/**
 * @brief   Convolve PSFs with Gaussian(sigma) for all facets, producing
 *          single-convolved and double-convolved (weighted mean) PSFs.
 * @details For each facet, batches over n_freq frequency channels:
 *          - conv_psf   = PSF * G(sigma)      [per-channel]
 *          - conv2_mean = wmean(PSF * G^2)    [weighted mean over channels]
 *
 *          Scale 0 (sigma == 0) is handled as a fast path: conv_psf is a
 *          device-to-device copy and conv2_mean is a weighted channel mean.
 *
 * @param[in]  conv           Convolution plans over the PSF grid: forward and backward
 *                            batched over n_freq, two backward plans (conv, conv^2).
 * @param[in]  psfs           PSFs, device, shape (n_facets, n_freq, psf_h, psf_w).
 * @param[in]  d_sigma        Gaussian sigma for this scale, device, shape (1,).
 * @param[in]  scale_idx      Index of the selected scale (0 = delta / no convolution).
 * @param[in]  weights        Per-channel weights, device, shape (n_freq,).
 * @param[out] out_conv_psf   Single-convolved PSFs, device,
 *                            shape (n_facets, n_freq, psf_h, psf_w), pre-allocated.
 * @param[out] out_conv2_mean Double-convolved weighted-mean PSFs, device,
 *                            shape (n_facets, psf_h, psf_w), pre-allocated.
 */
void convolve_psfs_with_scale_async(const linalg::convolve_ctx& conv, core::span4d<float> psfs,
                                    core::span1d<float> d_sigma, int scale_idx, core::span1d<float> weights,
                                    core::span4d<float> out_conv_psf, core::span3d<float> out_conv2_mean);

void convolve_psfs_with_scales_async(const linalg::convolve_ctx& conv, core::span4d<float> psfs,
                                     core::span1d<float> d_sigmas, core::span1d<float> weights,
                                     core::span5d<float> out_conv_psf, core::span4d<float> out_conv2_mean);

}  // namespace fast_deconv::scale
