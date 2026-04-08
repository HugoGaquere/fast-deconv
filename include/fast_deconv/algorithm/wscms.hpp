#pragma once

#include <fast_deconv/algorithm/detail/scale.cuh>
#include <fast_deconv/algorithm/detail/wscms_minor_loop.cuh>
#include <fast_deconv/algorithm/wscms_types.hpp>
#include <fast_deconv/core/logger.hpp>
#include <fast_deconv/core/resources.hpp>
#include <fast_deconv/core/span_types.hpp>

namespace fast_deconv::algorithm::wscms {

wscms_result run_wscms(core::resources& resources, core::device_span4d<float>& dirty,
                                     core::device_span2d<float>& mean_residual,
                                     const core::device_span6d<float>& psfs,
                                     const core::device_span4d<float>& psfs_2,
                                     const core::device_span4d<float>& jones_norm,
                                     const core::device_vect<float>& weights_freq,
                                     WSCMS_ctx& wscms_ctx, const scale_convole_ctx& scale_ctx,
                                     WSCMS_params params)
{
  log::set_level(spdlog::level::info);
  FD_LOG_INFO("run_wscms: dirty={} psfs={} n_scales={} max_iter={} peak_factor={}", dirty, psfs,
              params.n_scales, params.max_iteration, params.peak_factor);

  wscms_result result =
      detail::run_wscms(resources, dirty, mean_residual, psfs, psfs_2, jones_norm, weights_freq,
                        wscms_ctx, scale_ctx, params);

  FD_LOG_INFO("run_wscms: completed");
  return result;
}

class Wscms {
 public:
  Wscms(const core::device_span6d<float>& psfs, const core::device_span4d<float>& psfs_2,
        const core::device_span2d<float>& xdes, const core::device_span2d<bool>& scale_masks,
        const core::device_vect<float>& scale_sigmas, const core::host_vect<float>& scale_bias,
        const core::host_span2d<int>& map_pixel_facet, const core::host_span2d<float>& gains,
        int dirty_nrows, int dirty_ncols, float peak_factor, bool clean_negative, float fft_padding,
        int exec_device = 0)
      : ctx_{psfs, psfs_2, xdes, scale_masks, scale_sigmas, scale_bias, map_pixel_facet, gains},
        params_{
            .clean_negative = clean_negative,
            .peak_factor = peak_factor,
            .max_iteration = 0,
            .n_scales = static_cast<int>(scale_sigmas.size()),
        },
        convolve_ctx_(detail::make_scale_convolve_ctx(
            dirty_nrows, dirty_ncols, static_cast<int>(scale_sigmas.size()), fft_padding)),
        resources_(exec_device) {};

  wscms_result run(core::device_span4d<float>& dirty,
                                 core::device_span2d<float>& mean_residual,
                                 const core::device_span4d<float>& jones_norm,
                                 const core::device_vect<float>& weights_freq, int max_iterations)
  {
    params_.max_iteration = max_iterations;
    return run_wscms(resources_, dirty, mean_residual, ctx_.psfs, ctx_.psfs_2, jones_norm,
                     weights_freq, ctx_, convolve_ctx_, params_);
  }

  void set_peak_factor(float v) { params_.peak_factor = v; }
  float peak_factor() const { return params_.peak_factor; }

  void set_clean_negative(bool v) { params_.clean_negative = v; }
  bool clean_negative() const { return params_.clean_negative; }

 private:
  WSCMS_ctx ctx_;
  WSCMS_params params_;
  scale_convole_ctx convolve_ctx_;
  core::resources resources_;
};

}  // namespace fast_deconv::algorithm::wscms
