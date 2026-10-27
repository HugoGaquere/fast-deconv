#pragma once
#include <cmath>
#include <fast_deconv/core/region.hpp>
#include <fast_deconv/core/exec_ctx.hpp>
#include <fast_deconv/matrix/peak.hpp>

namespace fast_deconv::common {

/**
 * @brief residual -= gain * psf, with @p psf centered on @p peak_coords and clipped to the image.
 *
 * Pixels where criterion(residual) < @p threshold are left untouched; the defaults subtract everywhere.
 * Enqueued on @p ctx.
 *
 * @param[in]     ctx          Execution context.
 * @param[in,out] residual     Image, device, (nrow, ncol).
 * @param[in]     psf          Component, device, (psf_h, psf_w); its center (psf_h / 2, psf_w / 2) lands on the peak.
 * @param[in]     peak_coords  Pixel of @p residual the component is centered on.
 * @param[in]     gain         Scale applied to @p psf.
 * @param[in]     criterion    Ranks each residual pixel; its mask indexes @p residual.
 * @param[in]     threshold    Pixels ranked below this are not touched.
 */
void subtract_component(const core::exec_ctx& ctx, core::span2d<float> residual, core::span2d<const float> psf,
                        core::index2d peak_coords, float gain, matrix::peak_criterion criterion = {},
                        float threshold = -INFINITY);

/**
 * @brief residual[f] -= gain * spectral_coeffs[f] * psf[f] for every channel f, with each plane centered on
 *        @p peak_coords and clipped to the image. Enqueued on @p ctx.
 *
 * @param[in]     ctx              Execution context.
 * @param[in,out] residual         Residual cube, device, (n_freq, nrow, ncol).
 * @param[in]     psf              Per-channel component, device, (n_freq, psf_h, psf_w).
 * @param[in]     spectral_coeffs  Per-channel flux of the component, device, (n_freq,).
 * @param[in]     peak_coords      Pixel the component is centered on.
 * @param[in]     gain             Scale applied to every channel.
 */
void subtract_component(const core::exec_ctx& ctx, core::span3d<float> residual, core::span3d<const float> psf,
                        core::span1d<const float> spectral_coeffs, core::index2d peak_coords, float gain);

}  // namespace fast_deconv::common
