#pragma once

#include <fast_deconv/algorithm/detail/scale.cuh>
#include <fast_deconv/algorithm/detail/wscms_minor_loop.cuh>
#include <fast_deconv/algorithm/wscms_types.hpp>
#include <fast_deconv/core/logger.hpp>
#include <fast_deconv/core/resources.hpp>
#include <fast_deconv/core/span_types.hpp>

namespace fast_deconv::algorithm::wscms {

std::vector<sky_component> run_wscms(core::device_span4d<float>& dirty,
                                     core::device_span2d<float>& mean_residual,
                                     const core::device_span6d<float>& psfs,
                                     const core::device_span4d<float>& psfs_2, WSCMS_ctx& wscms_ctx,
                                     WSCMS_params params)
{
  FD_LOG_INFO("run_wscms: dirty={} psfs={} n_scales={} max_subminor_iter={} peak_factor={}", dirty,
              psfs, params.n_scales, params.max_subminor_iter, params.peak_factor);
  FD_LOG_DEBUG("run_wscms: beam_enable={} do_abs={} per_scale_mask={} padding={}",
               params.beam_enable, params.do_abs, params.per_scale_mask, params.padding);

  core::resources resources(0);
  std::vector<sky_component> components =
      detail::run_wscms(resources, dirty, mean_residual, psfs, psfs_2, wscms_ctx, params);

  FD_LOG_INFO("run_wscms: completed");
  return components;
}

}  // namespace fast_deconv::algorithm::wscms
