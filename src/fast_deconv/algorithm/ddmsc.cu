#include <fast_deconv/algorithm/ddmsc.hpp>

#include "fast_deconv/util/cuda_macros.hpp"

namespace fast_deconv::algorithm::ddmsc {

Ddmsc::Ddmsc(const core::host_span4d<float>& raw_psfs, const core::host_span2d<float>& xdes,
             const core::host_span2d<bool>& mask, const core::host_vect<float>& scale_sigmas,
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

ddmsc_result Ddmsc::run(core::host_span3d<float>& dirty, const core::host_span3d<float>& jones_norm,
                        const core::host_vect<float>& weights_freq)
{
  const core::stream_resources& stream = ctx_.compute_stream;

  // Copy the per-call inputs host->device.
  auto d_dirty = stream.alloc_mdcontainer_async<float>(dirty.extent(0), dirty.extent(1), dirty.extent(2));
  auto d_jones =
      stream.alloc_mdcontainer_async<float>(jones_norm.extent(0), jones_norm.extent(1), jones_norm.extent(2));
  auto d_weights = stream.alloc_mdcontainer_async<float>(weights_freq.extent(0));

  CHECK_CUDA(cudaMemcpyAsync(d_dirty.data_handle(), dirty.data_handle(), dirty.size() * sizeof(float),
                             cudaMemcpyHostToDevice, stream.cuda_stream));
  CHECK_CUDA(cudaMemcpyAsync(d_jones.data_handle(), jones_norm.data_handle(), jones_norm.size() * sizeof(float),
                             cudaMemcpyHostToDevice, stream.cuda_stream));
  CHECK_CUDA(cudaMemcpyAsync(d_weights.data_handle(), weights_freq.data_handle(), weights_freq.size() * sizeof(float),
                             cudaMemcpyHostToDevice, stream.cuda_stream));

  core::device_span3d<float> dirty_view = d_dirty;
  ddmsc_result result = run_ddmsc_cycles(ctx_, params_, dirty_view, d_jones, d_weights);

  // Copy the mutated residual back into the caller's host buffer (in/out).
  CHECK_CUDA(cudaMemcpyAsync(dirty.data_handle(), d_dirty.data_handle(), dirty.size() * sizeof(float),
                             cudaMemcpyDeviceToHost, stream.cuda_stream));
  stream.sync();
  return result;
}

}  // namespace fast_deconv::algorithm::ddmsc
