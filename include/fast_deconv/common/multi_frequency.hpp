#pragma once

#include <fast_deconv/core/region.hpp>
#include <fast_deconv/core/exec_ctx.hpp>

namespace fast_deconv::multi_frequency {

/**
 * @brief Fit the spectral model of the component at @p peak_coords and its per-channel flux.
 *
 * With SAX[f, o] = sqrt(jones_norm[f, peak]) * xdes[f, o] and y[f] = residual[f, peak]:
 *   - n_freq > n_order: weighted least squares (weights_freq), corrected so the band-weighted mean of the fitted
 *     per-channel flux equals that of y;
 *   - n_freq <= n_order: the minimum-norm exact fit, theta = SAX^T (SAX SAX^T)^-1 y.
 * Solved in double; common/detail/spectral_fit.hpp derives both branches. A singular system leaves NaN in both
 * outputs.
 *
 * Runs on @p ctx; on the CUDA backend it is enqueued, not synced.
 *
 * @param[in]  ctx                  Execution context.
 * @param[in]  residual             Residual cube, device, (n_freq, nrow, ncol); read at @p peak_coords only.
 * @param[in]  jones_norm           Jones normalization, device, (n_freq, nrow, ncol); read at @p peak_coords only.
 * @param[in]  weights_freq         Per-channel weights, device, (n_freq,).
 * @param[in]  xdes                 Spectral design matrix, device, (n_freq, n_order), n_order <= 8.
 * @param[in]  peak_coords          Pixel to fit.
 * @param[out] compact_coeffs_out   Intrinsic coefficients theta, device, (n_order,).
 * @param[out] coeffs_per_chan_out  Fitted apparent flux per channel, SAX theta, device, (n_freq,).
 * @throws std::invalid_argument If n_order exceeds 8.
 */
void fit_coefficients(const core::exec_ctx& ctx, core::span3d<const float> residual,
                      core::span3d<const float> jones_norm, core::span1d<const float> weights_freq,
                      core::span2d<const float> xdes, core::index2d peak_coords,
                      core::span1d<float> compact_coeffs_out, core::span1d<float> coeffs_per_chan_out);

}  // namespace fast_deconv::multi_frequency