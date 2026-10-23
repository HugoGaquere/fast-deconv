#include <fast_deconv/algorithm/ddmsc.hpp>

namespace fast_deconv::algorithm::ddmsc {

Ddmsc::Ddmsc(core::host_span4d<const float> raw_psfs, core::host_span2d<const float> xdes,
             core::host_span2d<const bool> mask, core::host_span1d<const float> scale_sigmas,
             core::host_span1d<const float> scale_bias, core::host_span2d<const int> map_pixel_facet,
             std::int64_t dirty_nrow, std::int64_t dirty_ncol, int exec_device)
    : ctx_(exec_device, raw_psfs, xdes, mask, scale_sigmas, scale_bias, map_pixel_facet, dirty_nrow, dirty_ncol),
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

ddmsc_result Ddmsc::run(core::host_span3d<float> dirty, core::host_span3d<const float> jones_norm,
                        core::host_span1d<const float> weights_freq)
{
  const core::exec_ctx& ctx = ctx_.compute_stream;

  auto d_dirty = ctx.stage(dirty);
  auto d_jones = ctx.stage(jones_norm);
  auto d_weights = ctx.stage(weights_freq);

  ddmsc_result result = run_ddmsc_cycles(ctx_, params_, d_dirty, d_jones, d_weights);

  ctx.unstage(d_dirty, dirty.data_handle());
  ctx.wait();
  return result;
}

}  // namespace fast_deconv::algorithm::ddmsc
