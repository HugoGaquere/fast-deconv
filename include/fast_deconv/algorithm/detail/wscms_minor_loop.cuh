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
                                     const core::device_span4d<float>& psfs_2,
                                     const core::device_span4d<float>& jones_norm,
                                     const core::device_vect<float>& weights_freq,
                                     WSCMS_ctx& wscms_ctx,
                                     const scale_convole_ctx& scale_ctx, WSCMS_params params)
{
  log::set_level(spdlog::level::debug);

  bool per_scale_mask = false; // TODO: FIX THAT
  const auto& stream_r = resources.get_stream_resources();
  const int n_scales = params.n_scales;
  const int dirty_nrows = dirty.extent(2);
  const int dirty_ncols = dirty.extent(3);
  const int npix = dirty_nrows * dirty_ncols;
  const int freq_scales_total = scale_ctx.freq_nrow * scale_ctx.freq_ncol * n_scales;

  float* scaled_mean_dirty = nullptr;
  CHECK_CUDA(cudaMalloc(reinterpret_cast<void**>(&scaled_mean_dirty), dirty_nrows * dirty_ncols * sizeof(float)));

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
                      params.clean_negative, per_scale_mask);
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
      resources, dirty, scaled_mean_dirty, psfs, psfs_2, jones_norm, weights_freq, sel.best_scale,
      wscms_ctx, params);

  return components;
}

}  // namespace fast_deconv::algorithm::wscms::detail
