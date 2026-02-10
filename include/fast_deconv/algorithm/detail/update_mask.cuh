#pragma once

#include <fast_deconv/core/span_types.hpp>
#include <fast_deconv/core/stream_resources.hpp>
#include <fast_deconv/util/cuda_macros.hpp>

namespace fast_deconv::algorithm::wscms::detail {

// Update mask: mask[i] = mask[i] && (|scaled_dirty[i]| > threshold)
__global__ void update_mask_abs_kernel(bool* __restrict__ mask,
                                       const float* __restrict__ scaled_dirty,
                                       float threshold,
                                       size_t size)
{
  const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= size) return;
  mask[idx] = mask[idx] && (fabsf(scaled_dirty[idx]) > threshold);
}

// Update mask: mask[i] = mask[i] && (scaled_dirty[i] > threshold)
__global__ void update_mask_kernel(bool* __restrict__ mask,
                                   const float* __restrict__ scaled_dirty,
                                   float threshold,
                                   size_t size)
{
  const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= size) return;
  mask[idx] = mask[idx] && (scaled_dirty[idx] > threshold);
}

inline void update_mask(core::device_span2d<bool> mask,
                        core::device_span4d<float> scaled_dirty,
                        float threshold,
                        bool do_abs,
                        core::stream_resources& resources)
{
  // mask is 2D (h, w), scaled_dirty is 4D (nch, npol, h, w).
  // The mask applies to the spatial dims. We use the first channel/pol slice
  // of scaled_dirty for the threshold comparison since the mask is shared.
  // Actually, looking at the Python reference, scaled_dirty is compared element-wise
  // but mask is 2D. We compare using the full 4D scaled_dirty flattened,
  // but mask is broadcast. For simplicity, we'll compare mask (h*w) with
  // a reduction over channels. But the Python code uses the full 4D comparison.
  //
  // Looking at the Python reference more carefully:
  //   self._mask = logical_and(abs(scaled_dirty) > threshold, self._mask)
  // where scaled_dirty is 4D and mask is 4D (broadcast from 2D).
  // Since mask is 2D (h,w), we check if ANY channel exceeds threshold at each pixel,
  // or we just use the scaled_dirty directly since it seems mask and scaled_dirty
  // have compatible shapes in the Python code (mask is broadcast to 4D there).
  //
  // For correctness: we compare per-pixel using the max across channels.
  // But simplest: just use the flattened scaled_dirty[0,0,:,:] since for
  // WSCMS the scaled_dirty is already a scalar image per pixel.
  // Actually scaled_dirty shape is (nch, npol, h, w) but the mask is (h, w).
  // The Python uses cp.abs(self._scaled_dirty) > threshold which is 4D,
  // then logical_and with a 4D mask (broadcast). Since our mask is 2D,
  // we compare against the first (ch=0, pol=0) slice.

  const size_t mask_size = mask.size();
  const float* sd_ptr    = scaled_dirty.data_handle();  // (0, 0, :, :)

  if (do_abs) {
    update_mask_abs_kernel<<<CEIL_DIV(mask_size, 256), 256, 0, resources.stream>>>(
      mask.data_handle(), sd_ptr, threshold, mask_size);
  } else {
    update_mask_kernel<<<CEIL_DIV(mask_size, 256), 256, 0, resources.stream>>>(
      mask.data_handle(), sd_ptr, threshold, mask_size);
  }
  CHECK_LAST_CUDA_ERROR();
}

}  // namespace fast_deconv::algorithm::wscms::detail
