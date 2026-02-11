#pragma once
#include <cuda_runtime.h>

#include <fast_deconv/algorithm/wscms_types.hpp>
#include <fast_deconv/core/span_types.hpp>
#include <fast_deconv/core/stream_resources.hpp>
#include <fast_deconv/linalg/pinv.cuh>
#include <fast_deconv/matrix/argmax.hpp>
#include <fast_deconv/util/cuda_macros.hpp>
#include <fast_deconv/util/mdspan_utils.hpp>

#include <vector>

#include <fmt/core.h>

// 1. Find peak inside scaled_dirty              (stream_1)
// 2. Copy peak_value to host                    (stream_1)
// 3. Compute threshold                          (host)
// 3. Update mask                                (stream_1)
// 5. Sync stream_1                              (host waits for peak_value)
// 6. Sync device (if needed for control flow)   (cudaDeviceSynchronize)
// 7. Loop until threshold condition met         (host loop)
// 7a. Unravel index                             (host)
// 7a. Compute region                            (host)
// 7b. Copy y from dirty                         (stream_1)
// 7c. Join streams on 7a+7b                     ()
// 7d. Launch clean_dirty                        (stream_1)
// 7e. Launch clean_scaled_dirty                 (stream_2)
// 7f. Argmax on updated scaled_dirty            (stream_2)
// 7g. Sync streams                              (cudaEvent/cudaStreamSynchronize)

namespace fast_deconv::algorithm::wscms::detail {
// ================================================================== //
//                        Utils
// ================================================================== //
struct roi_strided {
  int ax0, ay0, bx0, by0;  // Top-left in A and B
  int w, h;                // Overlap size
  int lda, ldb;            // A and B row stride in elements
};

// Compute aligned bounding boxes to extract overlapping sub-regions from
// arrays A and B, where B is centered at a given position within A.
// lda and ldb are computed as the array widths.
__host__ __device__ inline roi_strided compute_roi(
  int y_center, int x_center, int a_height, int a_width, int b_height, int b_width)
{
  // X axis
  const int b_x0       = x_center - b_width / 2;
  const int a_x0       = b_x0 > 0 ? b_x0 : 0;
  const int b_x_offset = a_x0 - b_x0;

  const int b_x1           = x_center + b_width / 2;
  const int a_x1_inclusive = b_x1 < a_width - 1 ? b_x1 : a_width - 1;
  const int b_x1_offset    = b_x1 - a_x1_inclusive;
  const int b_x1_aligned   = b_width - b_x1_offset;

  // Y axis
  const int b_y0       = y_center - b_height / 2;
  const int a_y0       = b_y0 > 0 ? b_y0 : 0;
  const int b_y_offset = a_y0 - b_y0;

  const int b_y1           = y_center + b_height / 2;
  const int a_y1_inclusive = b_y1 < a_height - 1 ? b_y1 : a_height - 1;
  const int b_y1_offset    = b_y1 - a_y1_inclusive;
  const int b_y1_aligned   = b_height - b_y1_offset;

  // Compute overlap dimensions
  const int w = b_x1_aligned - b_x_offset;
  const int h = b_y1_aligned - b_y_offset;

  return roi_strided{a_x0, a_y0, b_x_offset, b_y_offset, w, h, a_width, b_width};
}

__host__ __device__ inline auto unravel_index_2D(uint flat_index, uint width)
  -> std::pair<uint, uint>
{
  const uint y = flat_index / width;
  const uint x = flat_index % width;
  return {y, x};
}

// Extract spatial (y, x) from a 4D flat index: flat = ch*(npol*h*w) + pol*(h*w) + y*w + x
// Since h*w divides evenly into the higher dims, flat % (h*w) gives the spatial index.
__host__ __device__ inline auto unravel_spatial_index(uint flat_index,
                                                      uint plane_stride,
                                                      uint width) -> std::pair<uint, uint>
{
  const uint spatial_idx = flat_index % plane_stride;
  return {spatial_idx / width, spatial_idx % width};
}

// ================================================================== //
//                        Update Mask
// ================================================================== //
__global__ void update_mask_kernel(core::device_span4d<float> dirty,
                                   core::device_span4d<bool> mask,
                                   bool do_abs,
                                   float threshold)
{
  const auto i = blockIdx.x * blockDim.x + threadIdx.x;
  const auto j = blockIdx.y * blockDim.y + threadIdx.y;

  // Use gridDim.z for the combined k,l dimensions
  const auto kl = blockIdx.z;
  const auto k  = kl / mask.extent(3);
  const auto l  = kl % mask.extent(3);

  if (i >= mask.extent(0) || j >= mask.extent(1) || k >= mask.extent(2) || l >= mask.extent(3))
    return;

  if (do_abs)
    mask(i, j, k, l) = mask(i, j, k, l) && fabs(dirty(i, j, k, l)) > threshold;
  else
    mask(i, j, k, l) = mask(i, j, k, l) && dirty(i, j, k, l) > threshold;
}

void update_mask(core::device_span4d<float>& dirty,
                 core::device_span4d<bool>& mask,
                 bool do_abs,
                 float threshold,
                 core::stream_resources& resources)
{
  dim3 block(16, 16, 1);
  dim3 grid(CEIL_DIV(mask.extent(0), block.x),
            CEIL_DIV(mask.extent(1), block.y),
            mask.extent(2) * mask.extent(3));
  // update_mask_kernel<<<grid, block, 0, resources.stream>>>(dirty, mask, do_abs, threshold);
  CHECK_LAST_CUDA_ERROR();
}

// ================================================================== //
//                     Dirty - PSF
// ================================================================== //
__global__ void clean_dirty_kernel(float* __restrict__ dirty,
                                   const float* __restrict__ psf,
                                   const bool* __restrict__ mask,
                                   const float* __restrict__ xdes_pinv,
                                   const float* __restrict__ xdes,
                                   core::device_vect<float> sqrt_weights,
                                   roi_strided roi,
                                   float gain,
                                   uint n_freq,
                                   uint n_order,
                                   uint peak_y,
                                   uint peak_x,
                                   uint dirty_plane_stride,
                                   uint psf_plane_stride,
                                   ComponentEntry* entries,
                                   uint iter_idx,
                                   int scale_idx)
{
  // Shared memory layout: [wy(n_freq), compact_spec_coeffs(n_order), per_chan_spec_coeffs(n_freq)]
  extern __shared__ float smem[];
  float* wy                   = smem;
  float* compact_spec_coeffs  = smem + n_freq;
  float* per_chan_spec_coeffs = compact_spec_coeffs + n_order;

  const int tid        = threadIdx.y * blockDim.x + threadIdx.x;
  const int block_size = blockDim.x * blockDim.y;

  // wy[f] = sqrt_weights[f] * dirty[f, 0, peak_y, peak_x]
  for (uint f = tid; f < n_freq; f += block_size) {
    const uint idx = f * dirty_plane_stride + peak_y * roi.lda + peak_x;
    wy[f]          = sqrt_weights(f) * dirty[idx];
  }
  __syncthreads();

  // compact_spec_coeffs = xdes_pinv[n_order, n_freq] @ wy[n_freq]
  for (uint o = tid; o < n_order; o += block_size) {
    float sum = 0.0f;
    for (uint f = 0; f < n_freq; f++)
      sum += xdes_pinv[o * n_freq + f] * wy[f];
    compact_spec_coeffs[o] = sum;
  }
  __syncthreads();

  // per_chan_spec_coeffs = xdes[n_freq, n_order] @ compact_spec_coeffs[n_order]
  for (uint f = tid; f < n_freq; f += block_size) {
    float sum = 0.0f;
    for (uint o = 0; o < n_order; o++)
      sum += xdes[f * n_order + o] * compact_spec_coeffs[o];
    per_chan_spec_coeffs[f] = sum;
  }
  __syncthreads();

  // Store component entry (once per block, thread 0 of block 0)
  if (blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0 && tid == 0) {
    ComponentEntry& entry = entries[iter_idx];
    entry.x               = static_cast<int>(peak_x);
    entry.y               = static_cast<int>(peak_y);
    entry.scale_idx       = scale_idx;
    entry.gain            = gain;
    entry.n_coeffs        = static_cast<int>(n_order);
    for (uint o = 0; o < n_order && o < MAX_SPECTRAL_ORDER; o++)
      entry.coeffs[o] = compact_spec_coeffs[o];
  }

  const int local_x = blockIdx.x * blockDim.x + threadIdx.x;
  const int local_y = blockIdx.y * blockDim.y + threadIdx.y;
  const uint fp     = blockIdx.z;  // combined (freq, pol) index

  if (local_x >= roi.w || local_y >= roi.h) return;

  const int a_x = roi.ax0 + local_x;
  const int a_y = roi.ay0 + local_y;
  const int b_x = roi.bx0 + local_x;
  const int b_y = roi.by0 + local_y;

  const uint a_offset = fp * dirty_plane_stride + a_y * roi.lda + a_x;
  const uint b_offset = fp * psf_plane_stride + b_y * roi.ldb + b_x;

  if (mask[a_offset]) dirty[a_offset] -= psf[b_offset] * gain * per_chan_spec_coeffs[fp];
}

// compute A = A - gain * B within the ROI, where mask is true
// Grid: (ceil(roi.w/16), ceil(roi.h/16), n_freq_pol)
// dirty and psf are raw pointers; psf is pre-offset to the (scale_idx, facet_idx) slice.
__global__ void masked_axpy_naive_kernel(float* __restrict__ dirty,
                                         const float* __restrict__ psf,
                                         const bool* __restrict__ mask,
                                         roi_strided roi,
                                         float gain,
                                         uint dirty_plane_stride,
                                         uint psf_plane_stride)
{
  const int local_x = blockIdx.x * blockDim.x + threadIdx.x;
  const int local_y = blockIdx.y * blockDim.y + threadIdx.y;
  const uint fp     = blockIdx.z;  // combined (freq, pol) index

  if (local_x >= roi.w || local_y >= roi.h) return;

  const int a_x = roi.ax0 + local_x;
  const int a_y = roi.ay0 + local_y;
  const int b_x = roi.bx0 + local_x;
  const int b_y = roi.by0 + local_y;

  const uint a_offset = fp * dirty_plane_stride + a_y * roi.lda + a_x;
  const uint b_offset = fp * psf_plane_stride + b_y * roi.ldb + b_x;

  if (mask[a_offset]) dirty[a_offset] -= psf[b_offset] * gain;
}

void masked_axpy_naive(float* dirty_ptr,
                       const float* psf_slice_ptr,
                       const bool* mask_ptr,
                       roi_strided roi,
                       float gain,
                       uint n_freq_pol,
                       uint dirty_plane_stride,
                       uint psf_plane_stride,
                       cudaStream_t stream)
{
  dim3 block(16, 16, 1);
  dim3 grid(CEIL_DIV(roi.w, block.x), CEIL_DIV(roi.h, block.y), n_freq_pol);
  masked_axpy_naive_kernel<<<grid, block, 0, stream>>>(
    dirty_ptr, psf_slice_ptr, mask_ptr, roi, gain, dirty_plane_stride, psf_plane_stride);
  CHECK_LAST_CUDA_ERROR();
}

void clean_dirty(float* dirty_ptr,
                 const float* psf_slice_ptr,
                 const bool* mask_ptr,
                 const float* xdes_pinv_ptr,
                 const float* xdes_ptr,
                 core::device_vect<float> sqrt_weights,
                 roi_strided roi,
                 float gain,
                 uint n_freq,
                 uint n_order,
                 uint n_freq_pol,
                 uint peak_y,
                 uint peak_x,
                 uint dirty_plane_stride,
                 uint psf_plane_stride,
                 ComponentEntry* entries,
                 uint iter_idx,
                 int scale_idx,
                 cudaStream_t stream)
{
  dim3 block(16, 16, 1);
  dim3 grid(CEIL_DIV(roi.w, block.x), CEIL_DIV(roi.h, block.y), n_freq_pol);
  const uint smem_size = (2 * n_freq + n_order) * sizeof(float);
  clean_dirty_kernel<<<grid, block, smem_size, stream>>>(dirty_ptr,
                                                         psf_slice_ptr,
                                                         mask_ptr,
                                                         xdes_pinv_ptr,
                                                         xdes_ptr,
                                                         sqrt_weights,
                                                         roi,
                                                         gain,
                                                         n_freq,
                                                         n_order,
                                                         peak_y,
                                                         peak_x,
                                                         dirty_plane_stride,
                                                         psf_plane_stride,
                                                         entries,
                                                         iter_idx,
                                                         scale_idx);
  CHECK_LAST_CUDA_ERROR();
}

// ================================================================== //
//                     Minor Cycle Launcher
// ================================================================== //
std::vector<ComponentEntry> wscms_minor_cycle(core::device_span4d<float>& dirty,
                                              core::device_span4d<float>& scaled_dirty,
                                              core::device_span6d<float>& psfs,
                                              core::device_span6d<float>& psfs_2,
                                              core::device_span4d<bool>& mask,
                                              core::host_span2d<float>& gains,
                                              std::uint32_t scale_idx,
                                              const MinorCycleContext& ctx)
{
  core::stream_resources resources_1, resources_2;
  const auto stream_1 = resources_1.stream;
  const auto stream_2 = resources_2.stream;

  const uint n_freq       = static_cast<uint>(dirty.extent(0));
  const uint n_pol        = static_cast<uint>(dirty.extent(1));
  const uint dirty_height = static_cast<uint>(dirty.extent(2));
  const uint dirty_width  = static_cast<uint>(dirty.extent(3));

  const uint psf_height = static_cast<uint>(psfs.extent(4));
  const uint psf_width  = static_cast<uint>(psfs.extent(5));

  const uint n_freq_pol         = n_freq * n_pol;
  const uint dirty_plane_stride = dirty_height * dirty_width;
  const uint psf_plane_stride   = psf_height * psf_width;

  // Stride in the PSF 6D array to reach (scale_idx, facet_idx, 0, 0, 0, 0)
  // layout_right: stride for dim1 = e2*e3*e4*e5, stride for dim0 = e1*dim1
  const uint psf_facet_stride = n_freq_pol * psf_plane_stride;
  const uint psf_scale_stride = static_cast<uint>(psfs.extent(1)) * psf_facet_stride;

  // Compute pseudo-inverse of the design matrix: Xdes_pinv = pinv(Xdes)
  // Xdes is (n_freq x n_order), Xdes_pinv is (n_order x n_freq)
  const int n_order = static_cast<int>(ctx.Xdes.extent(1));
  float* d_Xdes_pinv_ptr;
  CHECK_CUDA(cudaMallocAsync(
    reinterpret_cast<void**>(&d_Xdes_pinv_ptr), n_order * n_freq * sizeof(float), stream_1));
  core::device_span2d<float> Xdes_pinv(d_Xdes_pinv_ptr, n_order, n_freq);
  linalg::pinv(ctx.Xdes, Xdes_pinv, n_freq, n_order, resources_1);

  ComponentEntry* d_entries;
  CHECK_CUDA(cudaMallocManaged(reinterpret_cast<void**>(&d_entries),
                               ctx.n_subminor_iter * sizeof(ComponentEntry)));

  float* dirty_spectral_peak_values;
  CHECK_CUDA(cudaMallocAsync(
    reinterpret_cast<void**>(&dirty_spectral_peak_values), n_freq * sizeof(float), stream_2));

  float* peak_value;
  uint* peak_idx;
  CHECK_CUDA(cudaMallocManaged(reinterpret_cast<void**>(&peak_value), sizeof(float)));
  CHECK_CUDA(cudaMallocManaged(reinterpret_cast<void**>(&peak_idx), sizeof(uint)));

  matrix::argmax_async(peak_value, peak_idx, scaled_dirty.data_handle(), mask.data_handle(),
                       scaled_dirty.size(), mask.size(), ctx.do_abs, resources_1);

  resources_1.sync();
  const float threshold = ctx.peak_factor * (*peak_value);

  fmt::print("[wscms] image: {}x{}, psf: {}x{}, max_iter: {}, threshold: {:.6e}, initial_peak: {:.6e}\n",
             dirty_width, dirty_height, psf_width, psf_height, ctx.n_subminor_iter, threshold, *peak_value);

  // mask = mask & dirty > threshold
  update_mask(dirty, mask, ctx.do_abs, threshold, resources_1);

  // Barrier sync before loop
  resources_1.sync(); resources_2.sync();

  uint n_iter = 0;
  while (*peak_value > threshold && n_iter < ctx.n_subminor_iter) {
    const auto [peak_y, peak_x] = unravel_spatial_index(*peak_idx, dirty_plane_stride, dirty_width);
    const roi_strided roi       = compute_roi(peak_y, peak_x, dirty_height, dirty_width, psf_height, psf_width);
    const int facet_idx         = ctx.map_pixels_facets(peak_y, peak_x);
    const float gain            = gains(scale_idx, facet_idx);

    fmt::print("[wscms] iter: {}/{}, peak: ({}, {}), value: {:.6e}, threshold: {:.6e}, roi: {}x{}\n",
               n_iter + 1, ctx.n_subminor_iter, peak_y, peak_x, *peak_value, threshold, roi.w, roi.h);

    // Pre-offset psf pointer to the (scale_idx, facet_idx) slice
    const float* psf_slice = psfs.data_handle() + scale_idx * psf_scale_stride + facet_idx * psf_facet_stride;

    clean_dirty(dirty.data_handle(), psf_slice, mask.data_handle(), d_Xdes_pinv_ptr,
                ctx.Xdes.data_handle(), ctx.sqrt_weights,
                roi, gain, n_freq, n_order, n_freq_pol, peak_y,
                peak_x, dirty_plane_stride, psf_plane_stride, d_entries,
                n_iter, static_cast<int>(scale_idx), stream_1);

    masked_axpy_naive(scaled_dirty.data_handle(), psf_slice, mask.data_handle(),
                      roi, gain, n_freq_pol, dirty_plane_stride, psf_plane_stride, stream_2);

    matrix::argmax_async(peak_value, peak_idx, scaled_dirty.data_handle(), mask.data_handle(),
                         scaled_dirty.size(), mask.size(), ctx.do_abs, resources_2);

    resources_1.sync(); resources_2.sync();
    n_iter++;
  }

  // Copy component entries to host vector before freeing
  std::vector<ComponentEntry> result(d_entries, d_entries + n_iter);

  CHECK_CUDA(cudaFreeAsync(d_Xdes_pinv_ptr, stream_1));
  CHECK_CUDA(cudaFreeAsync(dirty_spectral_peak_values, stream_2));
  CHECK_CUDA(cudaFree(d_entries));
  CHECK_CUDA(cudaFree(peak_value));
  CHECK_CUDA(cudaFree(peak_idx));
  CHECK_LAST_CUDA_ERROR();

  return result;
}

}  // namespace fast_deconv::algorithm::wscms::detail
