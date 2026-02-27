#pragma once

#include <fast_deconv/algorithm/detail/scale.cuh>
#include <fast_deconv/algorithm/detail/wscms_minor_loop.cuh>
#include <fast_deconv/algorithm/wscms_types.hpp>
#include <fast_deconv/core/span_types.hpp>
#include <vector>

namespace fast_deconv::algorithm::wscms {

std::vector<ComponentEntry> minor_cycle(core::device_span4d<float> dirty,
                                        core::device_span4d<float> scaled_dirty,
                                        core::device_span6d<float> psfs,
                                        core::device_span6d<float> psfs_2,
                                        core::device_span4d<bool> mask,
                                        core::host_span2d<float> gains, std::uint32_t scale_idx,
                                        const MinorCycleContext& ctx)
{
  return detail::minor_cycle(dirty, scaled_dirty, psfs, psfs_2, mask, gains, scale_idx, ctx);
}

void make_scales(core::device_vect<float> sigmas, core::device_span3d<float> out_scales, int scale_y_full) {
  const uint n_scales = sigmas.size();
  const uint scale_x = out_scales.extent(1);
  const uint scale_y = out_scales.extent(2);
  detail::make_scales(sigmas.data_handle(), scale_x, scale_y, scale_y_full, n_scales, out_scales.data_handle());
}

void scale_convolve(core::device_span2d<float> dirty, core::device_span3d<complex_type> scales,
                    core::device_span3d<float> out_scaled_dirty)
{
  // todo: perform some checks

  const uint dirty_x = dirty.extent(0);
  const uint dirty_y = dirty.extent(1);
  const uint n_scales = scales.extent(0);
  const uint scale_x = scales.extent(1);
  const uint scale_y = scales.extent(2);
  detail::scale_convolve(dirty.data_handle(), scales.data_handle(), out_scaled_dirty.data_handle(),
                         dirty_x, dirty_y, scale_x, scale_y, n_scales);
}

}  // namespace fast_deconv::algorithm::wscms
