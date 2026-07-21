#pragma once

#include <fast_deconv/algorithm/ddmsc_types.hpp>
#include <fast_deconv/core/resources.hpp>
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
ddmsc_result run_ddmsc_cycles(context& ctx, const params& p, core::device_span3d<float>& dirty,
                              const core::device_span3d<float>& jones_norm,
                              const core::device_vect<float>& weights_freq);

// {
//   log::set_level(spdlog::level::debug);
//   FD_LOG_INFO("run_ddmsc: dirty={} raw_psfs={} n_scales={}, max_iter={}, max_sub_iter={}, peak_factor={}", dirty,
//               ddmsc_ctx.raw_psfs, params.n_scales, params.max_iteration, params.max_sub_iteration,
//               params.peak_factor);
//   FD_LOG_DEBUG(
//       "run_ddmsc: jones_norm={} weights_freq_size={} gamma={:.6f} clean_negative={} "
//       "stop_flux={:.8f} divergence_factor={:.4f} stall_threshold={:.8f} forbidden_scales_count={}",
//       jones_norm, weights_freq.size(), params.gamma, params.clean_negative, params.stop_flux,
//       params.divergence_factor, params.stall_threshold, params.forbidden_scales.size());
//
//   ddmsc_result result = detail::ddmsc_minor_cycles(resources, dirty, jones_norm, weights_freq,
//                                                    ddmsc_ctx, scale_ctx, psf_ctx, mask, params);
//
//   FD_LOG_INFO("run_ddmsc: completed ({} iterations, exit={})", result.total_iterations,
//               static_cast<int>(result.exit_reason));
//   return result;
// }

// class Ddmsc {
//  public:
//   Ddmsc(const core::device_span5d<float>& raw_psfs, const core::device_span2d<float>& xdes,
//         const core::device_span2d<bool>& scale_mask, const core::device_vect<float>& scale_sigmas,
//         const core::host_vect<float>& scale_bias, const core::host_span2d<int>& map_pixel_facet,
//         float gamma, int dirty_nrows, int dirty_ncols, float peak_factor,
//         bool clean_negative, float fft_padding, int exec_device = 0)
//       : resources_(exec_device),
//         ctx_{raw_psfs, xdes, scale_mask, scale_sigmas, scale_bias, map_pixel_facet},
//         params_{
//             .clean_negative = clean_negative,
//             .peak_factor = peak_factor,
//             .gamma = gamma,
//             .max_sub_iteration = 0,
//             .n_scales = static_cast<int>(scale_sigmas.size()),
//         },
//         convolve_ctx_(detail::make_scale_convolve_ctx(
//             resources_, dirty_nrows, dirty_ncols, static_cast<int>(scale_sigmas.size()),
//             fft_padding)),
//         psf_convolve_ctx_(detail::make_psf_convolve_ctx(
//             resources_, static_cast<int>(raw_psfs.extent(3)),
//             static_cast<int>(raw_psfs.extent(4)), static_cast<int>(raw_psfs.extent(1)),
//             fft_padding)) {};
//
//   ddmsc_result run(core::device_span4d<float>& dirty,
//                    const core::device_span4d<float>& jones_norm,
//                    const core::device_vect<float>& weights_freq,
//                    const core::device_span2d<bool>& mask, float stop_flux, int max_iteration,
//                    int max_sub_iteration, float divergence_factor, float stall_threshold,
//                    const std::vector<int>& forbidden_scales)
//   {
//     params_.max_sub_iteration = max_sub_iteration;
//     params_.stop_flux = stop_flux;
//     params_.max_iteration = max_iteration;
//     params_.divergence_factor = divergence_factor;
//     params_.stall_threshold = stall_threshold;
//     params_.forbidden_scales = forbidden_scales;
//
//     return run_ddmsc(resources_, dirty, jones_norm, weights_freq, ctx_, convolve_ctx_,
//                      psf_convolve_ctx_, mask.data_handle(), params_);
//   }
//
//   void set_peak_factor(float v) { params_.peak_factor = v; }
//   float peak_factor() const { return params_.peak_factor; }
//
//   void set_clean_negative(bool v) { params_.clean_negative = v; }
//   bool clean_negative() const { return params_.clean_negative; }
//
//  private:
//   core::resources resources_;  // constructed first, destroyed last (pool owns workspace)
//   DDMSC_ctx ctx_;
//   DDMSC_params params_;
//   scale_convolve_ctx convolve_ctx_;
//   psf_convolve_ctx psf_convolve_ctx_;
// };

}  // namespace fast_deconv::algorithm::ddmsc
