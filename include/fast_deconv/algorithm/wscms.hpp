#pragma once

#include <fast_deconv/algorithm/detail/wscms_minor_loop.cuh>
#include <fast_deconv/algorithm/detail/scale.cuh>
#include <fast_deconv/algorithm/wscms_types.hpp>
#include <fast_deconv/core/logger.hpp>
#include <fast_deconv/core/resources.hpp>
#include <fast_deconv/core/span_types.hpp>

namespace fast_deconv::algorithm::wscms {

void run_wscms(core::device_span4d<float>& dirty, core::device_span2d<float>& mean_residual,
               const core::device_span6d<float>& psfs, const core::device_span4d<float>& psfs_2,
               WSCMS_ctx& wscms_ctx, WSCMS_params params)
{
  FD_LOG_INFO("run_wscms: dirty={} psfs={} n_scales={} max_subminor_iter={} peak_factor={}",
              dirty, psfs, params.n_scales, params.max_subminor_iter, params.peak_factor);
  FD_LOG_DEBUG("run_wscms: beam_enable={} do_abs={} per_scale_mask={} padding={}",
               params.beam_enable, params.do_abs, params.per_scale_mask, params.padding);

  core::resources resources(0);
  detail::run_wscms(resources, dirty, mean_residual, psfs, psfs_2, wscms_ctx, params);

  FD_LOG_INFO("run_wscms: completed");
}

// std::vector<ComponentEntry> minor_cycle(const core::resources&, const MinorCycleContext& ctx,
//                                         core::device_span4d<float> dirty,
//                                         core::device_span4d<float> scaled_dirty,
//                                         core::device_span6d<float> psfs,
//                                         core::device_span6d<float> psfs_2,
//                                         core::device_span4d<bool> mask,
//                                         core::host_span2d<float> gains, std::uint32_t scale_idx)
// {
//   return detail::minor_cycle(dirty, scaled_dirty, psfs, psfs_2, mask, gains, scale_idx, ctx);
// }

// void wscms_minor_cycles(const core::resources& resources, core::device_span2d<float>& residual,
//                         core::device_span4d<float>& psfs_2,
//                         core::device_span2d<int>& map_pixels_facets,
//                         core::device_span2d<float>& gains, int scale_idx, float threshold,
//                         int max_iter)
// {
  // if (residual.extent(0) != map_pixels_facets.extent(0) || residual.extent(1) !=
  // map_pixels_facets.extent(1))
  //   throw std::invalid_argument(
  //       "wscms_minor_cycles: residual and map_pixels_facets spatial "
  //       "dimensions must match");
  // if (scale_idx < 0 || scale_idx >= static_cast<int>(psfs_2.extent(0)))
  //   throw std::invalid_argument("wscms_minor_cycles: scale_idx out of range");
  // if (psfs_2.extent(1) != gains.extent(1))
  //   throw std::invalid_argument("wscms_minor_cycles: psfs_2 n_facet != gains n_facet");
  // if (threshold <= 0) throw std::invalid_argument("wscms_minor_cycles: threshold must be > 0");
  // if (max_iter <= 0) throw std::invalid_argument("wscms_minor_cycles: max_iter must be > 0");
  //
  // detail::wscms_minor_cycles(resources, residual, psfs_2, map_pixels_facets, gains, scale_idx,
  //                            threshold, max_iter);
// }

// void wscms_minor_cycles_host_loop(
//     const core::resources& resources, core::device_span4d<float>& residual,
//     core::device_span2d<float>& mean_residual, const core::device_span6d<float>& psfs,
//     const core::device_span4d<float>& psfs_2, const core::device_span4d<float>& jones_norm,
//     const core::device_span2d<float>& xdes, const core::device_vect<float>& weights,
//     const core::host_span2d<int>& map_pixels_facets, const core::host_span2d<float>& gains,
//     int scale_idx, float threshold, int max_iter)
// {
//   if (residual.extent(0) != map_pixels_facets.extent(0) ||
//       residual.extent(1) != map_pixels_facets.extent(1))
//     throw std::invalid_argument(
//         "wscms_minor_cycles_host_loop: residual and map_pixels_facets "
//         "spatial dimensions must match");
//   if (scale_idx < 0 || scale_idx >= static_cast<int>(psfs_2.extent(0)))
//     throw std::invalid_argument("wscms_minor_cycles_host_loop: scale_idx out of range");
//   if (psfs_2.extent(1) != gains.extent(1))
//     throw std::invalid_argument("wscms_minor_cycles_host_loop: psfs_2 n_facet != gains n_facet");
//   if (threshold <= 0)
//     throw std::invalid_argument("wscms_minor_cycles_host_loop: threshold must be > 0");
//   if (max_iter <= 0)
//     throw std::invalid_argument("wscms_minor_cycles_host_loop: max_iter must be > 0");
//
//   detail::wscms_minor_cycles_host_loop(resources, residual, mean_residual, psfs, psfs_2, jones_norm,
//                                        xdes, weights, map_pixels_facets, gains, scale_idx,
//                                        threshold, max_iter);
// }
//
// void make_scales(const core::resources& resources, core::device_vect<float> sigmas,
//                  core::device_span3d<float> out_scales, int scale_ncol_full)
// {
//   if (sigmas.size() != out_scales.extent(0))
//     throw std::invalid_argument("make_scales: sigmas.size() != out_scales n_scales");
//   if (scale_ncol_full <= 0) throw std::invalid_argument("make_scales: scale_ncol_full must be > 0");
//   if (out_scales.extent(2) != static_cast<std::size_t>(scale_ncol_full / 2 + 1))
//     throw std::invalid_argument("make_scales: out_scales ncol must be scale_ncol_full / 2 + 1");
//
//   const uint n_scales = sigmas.size();
//   const uint scale_nrow = out_scales.extent(1);
//   const uint scale_ncol = out_scales.extent(2);
//   detail::make_scales(resources, sigmas.data_handle(), scale_nrow, scale_ncol, scale_ncol_full,
//                       n_scales, out_scales.data_handle());
// }
//
// scale_convole_ctx make_scale_convolve_ctx(int nrow, int ncol, int n_scales, float padding)
// {
//   return detail::make_scale_convolve_ctx(nrow, ncol, n_scales, padding);
// }
//
// void scale_convolve(const core::resources& resources, const scale_convole_ctx& ctx,
//                     core::device_span2d<float> dirty, core::device_span3d<float> scales,
//                     core::device_span3d<float> out_scaled_dirty)
// {
//   if (static_cast<int>(dirty.extent(0)) != ctx.img_nrow ||
//       static_cast<int>(dirty.extent(1)) != ctx.img_ncol)
//     throw std::invalid_argument("scale_convolve: dirty dimensions must match ctx image size");
//   if (static_cast<int>(scales.extent(0)) != ctx.n_batches)
//     throw std::invalid_argument("scale_convolve: scales n_scales must match ctx.n_batches");
//   if (static_cast<int>(scales.extent(1)) != ctx.freq_nrow ||
//       static_cast<int>(scales.extent(2)) != ctx.freq_ncol)
//     throw std::invalid_argument("scale_convolve: scales freq dimensions must match ctx");
//   if (static_cast<int>(out_scaled_dirty.extent(0)) != ctx.n_batches)
//     throw std::invalid_argument(
//         "scale_convolve: out_scaled_dirty n_scales must match ctx.n_batches");
//   if (static_cast<int>(out_scaled_dirty.extent(1)) != ctx.img_nrow ||
//       static_cast<int>(out_scaled_dirty.extent(2)) != ctx.img_ncol)
//     throw std::invalid_argument(
//         "scale_convolve: out_scaled_dirty spatial dimensions must match ctx image size");
//
//   const uint n_scales = scales.extent(0);
//   detail::scale_convolve(resources, ctx, dirty.data_handle(), scales.data_handle(),
//                          out_scaled_dirty.data_handle(), n_scales);
// }
//
// // MaskSpan: device_span2d<bool> (shared mask) or device_span3d<bool> (per-scale mask)
// template <typename MaskSpan>
// scale_selection_result scale_selection(const core::resources& resources,
//                                        core::device_span3d<float> scaled_dirty, MaskSpan mask,
//                                        core::host_vect<float> bias, bool do_abs)
// {
//   static_assert(MaskSpan::rank() == 2 || MaskSpan::rank() == 3,
//                 "scale_selection: mask must be rank 2 (shared) or rank 3 (per-scale)");
//   constexpr bool per_scale_mask = (MaskSpan::rank() == 3);
//
//   const int n_scales = scaled_dirty.extent(0);
//   const int nrow = scaled_dirty.extent(1);
//   const int ncol = scaled_dirty.extent(2);
//
//   if (bias.size() != static_cast<std::size_t>(n_scales))
//     throw std::invalid_argument("scale_selection: bias.size() != n_scales");
//   if constexpr (per_scale_mask) {
//     if (mask.extent(0) != scaled_dirty.extent(0) || mask.extent(1) != scaled_dirty.extent(1) ||
//         mask.extent(2) != scaled_dirty.extent(2))
//       throw std::invalid_argument(
//           "scale_selection: per-scale mask dimensions must match scaled_dirty");
//   } else {
//     if (mask.extent(0) != scaled_dirty.extent(1) || mask.extent(1) != scaled_dirty.extent(2))
//       throw std::invalid_argument(
//           "scale_selection: shared mask spatial dimensions must match scaled_dirty");
//   }
//
//   return detail::scale_selection(resources, scaled_dirty.data_handle(), mask.data_handle(),
//                                  bias.data_handle(), n_scales, nrow, ncol, do_abs, per_scale_mask);
// }

}  // namespace fast_deconv::algorithm::wscms
