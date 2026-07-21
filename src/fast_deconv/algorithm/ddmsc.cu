#include <fast_deconv/algorithm/ddmsc.hpp>

namespace fast_deconv::algorithm::ddmsc {

Ddmsc::Ddmsc(const core::device_span4d<float>& raw_psfs, const core::device_span2d<float>& xdes,
             const core::device_span2d<bool>& mask, const core::device_vect<float>& scale_sigmas,
             const core::host_vect<float>& scale_bias, const core::host_span2d<int>& map_pixel_facet, int dirty_nrow,
             int dirty_ncol, int n_freq, float fft_padding, int exec_device)
    : ctx_(exec_device, raw_psfs, xdes, mask, scale_sigmas, scale_bias, map_pixel_facet, dirty_nrow, dirty_ncol, n_freq,
           fft_padding),
      params_{
          .max_iteration = 1000,
          .divergence_factor = 2.0f,
          .flux_threshold = 0.0f,
          .stop_rms_factor = 0.0f,
          .stop_peak_factor = 0.0f,
          .stop_cycle_factor = 0.0f,
          .stop_sidelobe_level = 0.0f,
          .clean_negative = false,
          .peak_factor = 0.15f,
          .gamma = 0.1f,
          .max_clean_iteration = 1000,
          .scale_stall_threshold = 1e-6f,
          .enable_auto_mask = false,
          .force_enable_auto_mask = false,
          .auto_mask_peak_threshold = std::nullopt,
          .auto_mask_rms_threshold = std::nullopt,
      }
{
}

ddmsc_result Ddmsc::run(core::device_span3d<float>& dirty, const core::device_span3d<float>& jones_norm,
                        const core::device_vect<float>& weights_freq)
{
  return run_ddmsc_cycles(ctx_, params_, dirty, jones_norm, weights_freq);
}

}  // namespace fast_deconv::algorithm::ddmsc
