#pragma once

#include <fast_deconv/algorithm/detail/scale.cuh>
// #include <fast_deconv/algorithm/detail/wscms_minor_loop.cuh>
// #include <fast_deconv/algorithm/wscms_types.hpp>
#include <fast_deconv/core/span_types.hpp>
#include <vector>
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

// void make_scales(core::device_vect<float> sigmas, core::device_span3d<float> out_scales, int scale_y_full) {
//   const uint n_scales = sigmas.size();
//   const uint scale_x = out_scales.extent(1);
//   const uint scale_y = out_scales.extent(2);
//   detail::make_scales(sigmas.data_handle(), scale_x, scale_y, scale_y_full, n_scales, out_scales.data_handle());
// }

void scale_convolve(core::device_span2d<float> dirty, core::device_vect<float> sigmas,
                    core::device_span3d<float> out_scaled_dirty, float padding)
{
  // todo: perform some checks
  const uint dirty_x = dirty.extent(0);
  const uint dirty_y = dirty.extent(1);
  const uint n_scales = sigmas.extent(0);

  core::resources resources(0);

  detail::scale_convolve(dirty.data_handle(), sigmas.data_handle(), out_scaled_dirty.data_handle(),
                         dirty_x, dirty_y, n_scales, padding, resources);
}

// Shared mask: mask is (npix_x, npix_y), same for all scales
detail::ScaleSelectionResult scale_selection(core::device_span3d<float> scaled_dirty,
                                             core::device_span2d<bool> mask,
                                             core::host_vect<float> bias, bool do_abs)
{
  const int n_scales = scaled_dirty.extent(0);
  const int npix_x = scaled_dirty.extent(1);
  const int npix_y = scaled_dirty.extent(2);

  core::resources resources(0);

  // cudaStream_t stream = NULL;
  // CHECK_CUDA(cudaStreamCreate(&stream));
  auto result = detail::scale_selection(scaled_dirty.data_handle(), mask.data_handle(),
                                        bias.data_handle(), n_scales, npix_x, npix_y, do_abs,
                                        false, resources);
  // CHECK_CUDA(cudaStreamSynchronize(stream));
  // CHECK_CUDA(cudaStreamDestroy(stream));
  return result;
}

// Per-scale mask: mask is (n_scales, npix_x, npix_y), one per scale
detail::ScaleSelectionResult scale_selection(core::device_span3d<float> scaled_dirty,
                                             core::device_span3d<bool> mask,
                                             core::host_vect<float> bias, bool do_abs)
{
  const int n_scales = scaled_dirty.extent(0);
  const int npix_x = scaled_dirty.extent(1);
  const int npix_y = scaled_dirty.extent(2);
  core::resources resources(0);

  // cudaStream_t stream = NULL;
  // CHECK_CUDA(cudaStreamCreate(&stream));
  auto result = detail::scale_selection(scaled_dirty.data_handle(), mask.data_handle(),
                                        bias.data_handle(), n_scales, npix_x, npix_y, do_abs,
                                        true, resources);
  // CHECK_CUDA(cudaStreamSynchronize(stream));
  // CHECK_CUDA(cudaStreamDestroy(stream));
  return result;
}

}  // namespace fast_deconv::algorithm::wscms
