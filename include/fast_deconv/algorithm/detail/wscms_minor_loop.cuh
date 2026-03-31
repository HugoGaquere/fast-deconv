#pragma once
#include <cooperative_groups.h>
#include <cuda_runtime.h>

#include <cfloat>
#include <cub/cub.cuh>
#include <fast_deconv/algorithm/wscms_types.hpp>
#include <fast_deconv/core/logger.hpp>
#include <fast_deconv/core/resources.hpp>
#include <fast_deconv/core/span_types.hpp>
#include <fast_deconv/util/cuda_macros.hpp>
#include <fast_deconv/util/mdspan_utils.hpp>

#include "clean_minor_loop.cuh"
#include "scale.cuh"

namespace fast_deconv::algorithm::wscms::detail {

std::vector<sky_component> run_wscms(const core::resources& resources,
                                     core::device_span4d<float>& dirty,
                                     core::device_span2d<float>& mean_residual,
                                     const core::device_span6d<float>& psfs,
                                     const core::device_span4d<float>& psfs_2, WSCMS_ctx wscms_ctx,
                                     WSCMS_params params)
{
  log::set_level(spdlog::level::debug);

  const int dirty_nrows = dirty.extent(2);
  const int dirty_ncols = dirty.extent(3);
  const scale_convole_ctx scale_ctx =
      make_scale_convolve_ctx(dirty_nrows, dirty_ncols, params.n_scales, params.padding);

  float* scaled_mean_dirty = nullptr;
  CHECK_CUDA(cudaMalloc(reinterpret_cast<void**>(&scaled_mean_dirty),
                        dirty_nrows * dirty_ncols * sizeof(float)));
  FD_LOG_DEBUG("scale_convolve_ctx: img=[{},{}] padded=[{},{}] freq=[{},{}] n_batches={}",
               scale_ctx.img_nrow, scale_ctx.img_ncol, scale_ctx.img_padded_nrow,
               scale_ctx.img_padded_ncol, scale_ctx.freq_nrow, scale_ctx.freq_ncol,
               scale_ctx.n_batches);

  const auto& stream_r = resources.get_stream_resources();
  const int n_scales = params.n_scales;
  const int npix = dirty_nrows * dirty_ncols;
  const int freq_scales_total = scale_ctx.freq_nrow * scale_ctx.freq_ncol * n_scales;

  // 1. Generate Gaussian scale kernels in half-complex frequency domain
  float* scale_kernels = resources.alloc_async<float>(freq_scales_total, stream_r);
  make_scales(resources, stream_r, wscms_ctx.scale_sigmas.data_handle(), scale_ctx.freq_nrow,
              scale_ctx.freq_ncol, scale_ctx.img_padded_ncol, n_scales, scale_kernels);

  // 2. Convolve dirty image with all scale kernels
  float* scales_x_dirty = resources.alloc_async<float>(npix * n_scales, stream_r);
  scale_convolve(resources, stream_r, scale_ctx, mean_residual.data_handle(), scale_kernels,
                 scales_x_dirty, n_scales);

  // 3. Select the best scale
  scale_selection_result sel =
      scale_selection(resources, stream_r, scales_x_dirty, wscms_ctx.scale_masks.data_handle(),
                      wscms_ctx.scale_bias.data_handle(), n_scales, dirty_nrows, dirty_ncols,
                      params.do_abs, params.per_scale_mask);
  stream_r.sync();

  // 4. Copy the winning slice to output
  copy_scale_slice(stream_r, scales_x_dirty, scaled_mean_dirty, sel.best_scale, npix);

  // 5. Cleanup
  resources.free_async(scales_x_dirty, stream_r);
  resources.free_async(scale_kernels, stream_r);
  stream_r.sync();

  FD_LOG_INFO("selected scale_idx={} peak={:.6f} at ({},{})", sel.best_scale, sel.best_peak,
              sel.best_row, sel.best_col);

  const std::vector<sky_component> components = wscms_minor_cycles_host_loop(
      resources, dirty, scaled_mean_dirty, psfs, psfs_2, sel.best_scale, wscms_ctx, params);

  return components;
}

}  // namespace fast_deconv::algorithm::wscms::detail
