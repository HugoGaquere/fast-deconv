#pragma once

#include <fast_deconv/algorithm/detail/scale.cuh>
#include <fast_deconv/algorithm/detail/wscms_minor_loop.cuh>
#include <fast_deconv/algorithm/wscms_types.hpp>
#include <fast_deconv/core/logger.hpp>
#include <fast_deconv/core/resources.hpp>
#include <fast_deconv/core/span_types.hpp>

namespace fast_deconv::algorithm::wscms {

wscms_result run_wscms(core::resources& resources, core::device_span4d<float>& dirty,
                       const core::device_span4d<float>& jones_norm,
                       const core::device_vect<float>& weights_freq,
                       WSCMS_ctx& wscms_ctx, const scale_convole_ctx& scale_ctx,
                       const psf_convolve_ctx& psf_ctx,
                       const bool* mask, WSCMS_params params)
{
  log::set_level(spdlog::level::info);
  FD_LOG_INFO("run_wscms: dirty={} raw_psfs={} n_scales={} max_iter={} peak_factor={}", dirty,
              wscms_ctx.raw_psfs, params.n_scales, params.max_sub_iteration, params.peak_factor);

  wscms_result result = detail::wscms_minor_cycles(resources, dirty, jones_norm, weights_freq,
                                                   wscms_ctx, scale_ctx, psf_ctx, mask, params);

  FD_LOG_INFO("run_wscms: completed ({} iterations, exit={})", result.total_iterations,
              static_cast<int>(result.exit_reason));
  return result;
}

class Wscms {
 public:
  Wscms(const core::device_span5d<float>& raw_psfs, const core::device_span2d<float>& xdes,
        const core::device_span2d<bool>& scale_masks, const core::device_vect<float>& scale_sigmas,
        const core::host_vect<float>& scale_bias, const core::host_span2d<int>& map_pixel_facet,
        float gamma, int dirty_nrows, int dirty_ncols, float peak_factor,
        bool clean_negative, float fft_padding, int exec_device = 0)
      : ctx_{raw_psfs, xdes, scale_masks, scale_sigmas, scale_bias, map_pixel_facet},
        params_{
            .clean_negative = clean_negative,
            .peak_factor = peak_factor,
            .gamma = gamma,
            .max_sub_iteration = 0,
            .n_scales = static_cast<int>(scale_sigmas.size()),
        },
        convolve_ctx_(detail::make_scale_convolve_ctx(
            dirty_nrows, dirty_ncols, static_cast<int>(scale_sigmas.size()), fft_padding)),
        psf_convolve_ctx_(detail::make_psf_convolve_ctx(
            static_cast<int>(raw_psfs.extent(3)), static_cast<int>(raw_psfs.extent(4)),
            static_cast<int>(raw_psfs.extent(1)), fft_padding)),
        resources_(exec_device) {};

  wscms_result run(core::device_span4d<float>& dirty,
                   const core::device_span4d<float>& jones_norm,
                   const core::device_vect<float>& weights_freq,
                   const core::device_span2d<bool>& mask, float stop_flux, int max_iteration,
                   int max_sub_iteration, float divergence_factor, float stall_threshold,
                   const std::vector<int>& forbidden_scales)
  {
    params_.max_sub_iteration = max_sub_iteration;
    params_.stop_flux = stop_flux;
    params_.max_iteration = max_iteration;
    params_.divergence_factor = divergence_factor;
    params_.stall_threshold = stall_threshold;
    params_.forbidden_scales = forbidden_scales;

    return run_wscms(resources_, dirty, jones_norm, weights_freq, ctx_, convolve_ctx_,
                     psf_convolve_ctx_, mask.data_handle(), params_);
  }

  void set_peak_factor(float v) { params_.peak_factor = v; }
  float peak_factor() const { return params_.peak_factor; }

  void set_clean_negative(bool v) { params_.clean_negative = v; }
  bool clean_negative() const { return params_.clean_negative; }

 private:
  WSCMS_ctx ctx_;
  WSCMS_params params_;
  scale_convole_ctx convolve_ctx_;
  psf_convolve_ctx psf_convolve_ctx_;
  core::resources resources_;
};

}  // namespace fast_deconv::algorithm::wscms
