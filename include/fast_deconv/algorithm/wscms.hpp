#pragma once

#include <fast_deconv/algorithm/detail/scale.cuh>
// #include <fast_deconv/algorithm/detail/wscms_minor_loop.cuh>
// #include <fast_deconv/algorithm/wscms_types.hpp>
#include <fast_deconv/core/span_types.hpp>
#include <vector>

#include "fast_deconv/algorithm/wscms_types.hpp"
#include "fast_deconv/core/resources.hpp"

namespace fast_deconv::algorithm::wscms {

// std::vector<ComponentEntry> minor_cycle(core::device_span4d<float> dirty,
//                                         core::device_span4d<float> scaled_dirty,
//                                         core::device_span6d<float> psfs,
//                                         core::device_span6d<float> psfs_2,
//                                         core::device_span4d<bool> mask,
//                                         core::host_span2d<float> gains, std::uint32_t scale_idx,
//                                         const MinorCycleContext& ctx)
// {
//   return detail::minor_cycle(dirty, scaled_dirty, psfs, psfs_2, mask, gains, scale_idx, ctx);
// }

void make_scales(core::device_vect<float> sigmas, core::device_span3d<float> out_scales,
                 int scale_ncol_full)
{
  const uint n_scales = sigmas.size();
  const uint scale_nrow = out_scales.extent(1);
  const uint scale_ncol = out_scales.extent(2);
  detail::make_scales(sigmas.data_handle(), scale_nrow, scale_ncol, scale_ncol_full, n_scales,
                      out_scales.data_handle());
}

scale_convole_ctx make_scale_convole_ctx(int nrow, int ncol, int n_scales, float padding) {
  return detail::make_scale_convole_ctx(nrow, ncol, n_scales, padding);
}

void scale_convolve(core::device_span2d<float> dirty, core::device_vect<float> sigmas,
                    core::device_span3d<float> out_scaled_dirty, float padding)
{
  // todo: perform some checks
  const uint nrow = dirty.extent(0);
  const uint ncol = dirty.extent(1);
  const uint n_scales = sigmas.extent(0);

  core::resources resources(0);
  scale_convole_ctx ctx = detail::make_scale_convole_ctx(nrow, ncol, n_scales, padding);

  detail::scale_convolve(resources, ctx, dirty.data_handle(), sigmas.data_handle(),
                         out_scaled_dirty.data_handle(), n_scales);
}

// MaskSpan: device_span2d<bool> (shared mask) or device_span3d<bool> (per-scale mask)
template <typename MaskSpan>
detail::ScaleSelectionResult scale_selection(core::device_span3d<float> scaled_dirty, MaskSpan mask,
                                             core::host_vect<float> bias, bool do_abs)
{
  constexpr bool per_scale_mask = (MaskSpan::rank() == 3);

  const int n_scales = scaled_dirty.extent(0);
  const int nrow = scaled_dirty.extent(1);
  const int ncol = scaled_dirty.extent(2);

  core::resources resources(0);

  return detail::scale_selection(resources, scaled_dirty.data_handle(), mask.data_handle(),
                                 bias.data_handle(), n_scales, nrow, ncol, do_abs,
                                 per_scale_mask);
}

}  // namespace fast_deconv::algorithm::wscms
