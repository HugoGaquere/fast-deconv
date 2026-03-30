#pragma once
#include <cooperative_groups.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cfloat>
#include <cub/cub.cuh>
#include <fast_deconv/algorithm/wscms_types.hpp>
#include <fast_deconv/core/logger.hpp>
#include <fast_deconv/core/resources.hpp>
#include <fast_deconv/core/span_types.hpp>
#include <fast_deconv/util/cuda_macros.hpp>
#include <fast_deconv/util/mdspan_utils.hpp>

namespace fast_deconv::algorithm::wscms::detail {

// ================================================================== //
//                        Utils
// ================================================================== //
struct overlap_region {
  int ax0, ay0, bx0, by0;  // Top-left in A and B
  int w, h;                // Overlap size
  int lda, ldb;            // A and B row stride in elements
};

/**
 * @brief   Compute aligned bounding boxes.
 * @details Compute aligned bounding boxes to extract overlapping sub-regions from
 *          arrays A and B, where B is centered at a given position within A.
 *          lda and ldb are computed as the array widths.
 *
 * @param[in] y_center
 * @param[in] x_center
 * @param[in] a_height
 * @param[in] a_width
 * @param[in] b_height
 * @param[in] b_width
 *
 * @returns overlap_region
 */
__host__ __device__ inline overlap_region compute_overlap_region(int y_center, int x_center,
                                                                 int a_height, int a_width,
                                                                 int b_height, int b_width)
{
  // X axis
  const int b_x0 = x_center - b_width / 2;
  const int a_x0 = b_x0 > 0 ? b_x0 : 0;
  const int b_x_offset = a_x0 - b_x0;

  const int b_x1 = x_center + b_width / 2;
  const int a_x1_inclusive = b_x1 < a_width - 1 ? b_x1 : a_width - 1;
  const int b_x1_offset = b_x1 - a_x1_inclusive;
  const int b_x1_aligned = b_width - b_x1_offset;

  // Y axis
  const int b_y0 = y_center - b_height / 2;
  const int a_y0 = b_y0 > 0 ? b_y0 : 0;
  const int b_y_offset = a_y0 - b_y0;

  const int b_y1 = y_center + b_height / 2;
  const int a_y1_inclusive = b_y1 < a_height - 1 ? b_y1 : a_height - 1;
  const int b_y1_offset = b_y1 - a_y1_inclusive;
  const int b_y1_aligned = b_height - b_y1_offset;

  // Compute overlap dimensions
  const int w = b_x1_aligned - b_x_offset;
  const int h = b_y1_aligned - b_y_offset;

  return overlap_region{a_x0, a_y0, b_x_offset, b_y_offset, w, h, a_width, b_width};
}

__host__ __device__ inline auto unravel_index_2D(uint flat_index, uint width)
    -> std::pair<uint, uint>
{
  const uint y = flat_index / width;
  const uint x = flat_index % width;
  return {y, x};
}

// ================================================================== //
//              Cooperative-kernel minor cycles
// ================================================================== //

namespace cg = cooperative_groups;

struct IndexedValue {
  float value;
  int index;
};

struct ArgMaxOp {
  __device__ __forceinline__ IndexedValue operator()(const IndexedValue& a,
                                                     const IndexedValue& b) const
  {
    return (a.value >= b.value) ? a : b;
  }
};

constexpr int COOP_BLOCK_SIZE = 256;

template <int BLOCK_SIZE>
__device__ void grid_argmax(
    cg::grid_group grid, IndexedValue thread_data, IndexedValue* __restrict__ block_scratch,
    typename cub::BlockReduce<IndexedValue, BLOCK_SIZE>::TempStorage& temp_storage)
{
  using BlockReduce = cub::BlockReduce<IndexedValue, BLOCK_SIZE>;

  auto block = cg::this_thread_block();
  const int tid = block.thread_rank();

  // Phase 1: per-block reduction
  IndexedValue block_max = BlockReduce(temp_storage).Reduce(thread_data, ArgMaxOp());

  if (tid == 0) {
    block_scratch[blockIdx.x] = block_max;
  }

  grid.sync();

  // Phase 2: block 0 reduces across blocks
  if (blockIdx.x == 0) {
    IndexedValue local{-FLT_MAX, -1};
    const auto num_blocks = grid.num_blocks();
    for (unsigned i = tid; i < num_blocks; i += block.num_threads()) {
      local = ArgMaxOp{}(local, block_scratch[i]);
    }

    IndexedValue final_max = BlockReduce(temp_storage).Reduce(local, ArgMaxOp());

    if (tid == 0) {
      block_scratch[0] = final_max;
    }
  }

  grid.sync();
}

// residual: device float nrow x ncol
// psfs: device float n_facet x psf_nrow x psf_ncol
// map_pixels_facets: device int nrow x ncol
// gains: device float n_facet
template <int BLOCK_SIZE>
__global__ void clean_minor_cycles_kernel(float* residual, float* psfs, int* map_pixels_facets,
                                          float* gains, int nrow, int ncol, int psf_nrow,
                                          int psf_ncol, int n_facet,
                                          IndexedValue* __restrict__ block_scratch, float threshold,
                                          int max_iter)
{
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  const int stride = blockDim.x * gridDim.x;
  const int n = nrow * ncol;

  auto grid = cg::this_grid();

  using BlockReduce = cub::BlockReduce<IndexedValue, BLOCK_SIZE>;
  __shared__ typename BlockReduce::TempStorage temp_storage;

  // Grid-stride loop: each thread reduces its chunk locally first
  IndexedValue my_data{-FLT_MAX, -1};
  for (int i = tid; i < n; i += stride) {
    if (residual[i] > my_data.value) {
      my_data = {residual[i], i};
    }
  }
  grid_argmax<BLOCK_SIZE>(grid, my_data, block_scratch, temp_storage);

  IndexedValue peak = block_scratch[0];

  int n_iter = 0;
  while (peak.value > threshold && n_iter < max_iter) {
    // Phase 1: Unravel peak index + compute ROI + gain factor
    // All threads compute identically (broadcast reads from L2)
    auto [peak_row, peak_col] = unravel_index_2D(peak.index, ncol);
    int facet_id = map_pixels_facets[peak_row * ncol + peak_col];
    overlap_region ovr = compute_overlap_region(peak_row, peak_col, nrow, ncol, psf_nrow, psf_ncol);
    float factor = gains[facet_id] * peak.value;
    const float* psf_facet = psfs + facet_id * psf_nrow * psf_ncol;

    // Phase 2: Grid-stride PSF subtraction
    int roi_size = ovr.w * ovr.h;
    for (int i = tid; i < roi_size; i += stride) {
      int local_y = i / ovr.w;
      int local_x = i % ovr.w;
      int a_offset = (ovr.ay0 + local_y) * ovr.lda + (ovr.ax0 + local_x);
      int b_offset = (ovr.by0 + local_y) * ovr.ldb + (ovr.bx0 + local_x);
      float r = residual[a_offset];
      if (r > -FLT_MAX)  // skip masked pixels
        residual[a_offset] = r - psf_facet[b_offset] * factor;
    }

    // Phase 3: Sync + full re-scan argmax
    grid.sync();

    IndexedValue my_data_iter{-FLT_MAX, -1};
    for (int i = tid; i < n; i += stride) {
      if (residual[i] > my_data_iter.value) {
        my_data_iter = {residual[i], i};
      }
    }
    grid_argmax<BLOCK_SIZE>(grid, my_data_iter, block_scratch, temp_storage);

    peak = block_scratch[0];
    n_iter++;
  }
}

// psfs_2 layout: (n_scale, n_facet, psf_nrow, psf_ncol)
void wscms_minor_cycles(const core::resources& resources, core::device_span2d<float>& residual,
                        core::device_span4d<float>& psfs_2,
                        core::device_span2d<int>& map_pixels_facets,
                        core::device_span2d<float>& gains, int scale_idx, float threshold,
                        int max_iter)
{
  int nrow = residual.extent(0);
  int ncol = residual.extent(1);
  int n_facet = psfs_2.extent(1);
  int psf_nrow = psfs_2.extent(2);
  int psf_ncol = psfs_2.extent(3);

  float* d_residual = residual.data_handle();
  float* d_psfs = psfs_2.data_handle() + scale_idx * n_facet * psf_nrow * psf_ncol;
  int* d_map = map_pixels_facets.data_handle();
  float* d_gains = gains.data_handle() + scale_idx * n_facet;

  const int n = nrow * ncol;
  const int num_blocks = CEIL_DIV(n, COOP_BLOCK_SIZE);

  int dev = 0, num_sms = 0, max_bpsm = 0;
  CHECK_CUDA(cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, dev));
  CHECK_CUDA(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &max_bpsm, clean_minor_cycles_kernel<COOP_BLOCK_SIZE>, COOP_BLOCK_SIZE, 0));

  const int launch_blocks = std::min(num_blocks, num_sms * max_bpsm);

  const auto& stream_r = resources.get_stream_resources();
  IndexedValue* d_block_scratch = resources.alloc_async<IndexedValue>(launch_blocks, stream_r);

  void* args[] = {static_cast<void*>(&d_residual), static_cast<void*>(&d_psfs),
                  static_cast<void*>(&d_map),      static_cast<void*>(&d_gains),
                  static_cast<void*>(&nrow),       static_cast<void*>(&ncol),
                  static_cast<void*>(&psf_nrow),   static_cast<void*>(&psf_ncol),
                  static_cast<void*>(&n_facet),    static_cast<void*>(&d_block_scratch),
                  static_cast<void*>(&threshold),  static_cast<void*>(&max_iter)};

  CHECK_CUDA(cudaLaunchCooperativeKernel((void*)clean_minor_cycles_kernel<COOP_BLOCK_SIZE>,
                                         dim3(launch_blocks), dim3(COOP_BLOCK_SIZE), args, 0,
                                         stream_r.cuda_stream));

  stream_r.sync();
  resources.free_async(d_block_scratch, stream_r);
}

// ================================================================== //
//     Host-loop minor cycles
// ================================================================== //

/**
 * @brief   Set pixels at or below threshold to -INFINITY in-place.
 * @details Matches the Python reference mask update before the minor loop:
 *          `self._mask = logical_and(abs(scaled_dirty) > threshold, self._mask)`
 *          Since masked pixels are already -INFINITY, this only needs to threshold
 *          the remaining finite values.
 */
__global__ void apply_threshold_mask_kernel(float* data, int n, float threshold)
{
  for (int tid = blockIdx.x * blockDim.x + threadIdx.x; tid < n; tid += blockDim.x * gridDim.x) {
    if (data[tid] <= threshold) data[tid] = -INFINITY;
  }
}

/**
 * @brief   Subtract PSF scaled by factor from residual on the specified overlap region.
 * @details Performs @p residual(ovr.a) -= @p factor * @p psf(ovr.b) element-wise over
 *          the overlap region defined by @p ovr. Uses a grid-stride loop so that
 *          any launch configuration covers the full region.
 *
 * @param[in,out] residual  Residual image, row-major with leading dimension @c ovr.lda.
 * @param[in]     psf       PSF image, row-major with leading dimension @c ovr.ldb.
 * @param[in]     ovr       Overlap region descriptor giving the origin and extent of
 *                          the sub-windows in both @p residual (a) and @p psf (b).
 * @param[in]     factor    Loop gain (scale factor applied to the PSF before subtraction).
 */
__global__ void psf_subtract_kernel(float* __restrict__ residual, const float* __restrict__ psf,
                                    overlap_region ovr, float factor)
{
  const int n = ovr.w * ovr.h;
  for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < n; idx += blockDim.x * gridDim.x) {
    const int local_y = idx / ovr.w;
    const int local_x = idx % ovr.w;

    const int a_offset = (ovr.ay0 + local_y) * ovr.lda + (ovr.ax0 + local_x);
    const int b_offset = (ovr.by0 + local_y) * ovr.ldb + (ovr.bx0 + local_x);

    residual[a_offset] -= psf[b_offset] * factor;
  }
}

// Compute A[f, o] = xdes[f, o] * sqrt(jones_norm[f, peak_row, peak_col]) * sqrt(weights[f])
// jones_norm row-major: [n_freq, nrow, ncol]
// xdes row-major:       [n_freq, n_order]
// weights row-major:    [n_freq]
// A row-major:          [n_freq, n_order]
__global__ void compute_spectral_matrix_kernel(float* __restrict__ A,
                                               const float* __restrict__ xdes,
                                               const float* __restrict__ jones_norm,
                                               const float* __restrict__ weights, int n_freq,
                                               int n_order, int nrow, int ncol, int peak_row,
                                               int peak_col)
{
  const int n = n_freq * n_order;
  const int spatial_stride = nrow * ncol;
  const int peak_offset = peak_row * ncol + peak_col;
  for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < n; idx += blockDim.x * gridDim.x) {
    const int f = idx / n_order;
    float jn = jones_norm[f * spatial_stride + peak_offset];
    A[idx] = xdes[idx] * sqrtf(jn) * sqrtf(weights[f]);
  }
}

// Compute A_pinv = inv(A^T A) @ A^T
// A: row-major [n_freq, n_order] (device)
// A_pinv: col-major [n_order, n_freq] output (device), pre-allocated
//
// cuBLAS sees row-major A as col-major A_cm = A^T [n_order, n_freq]
//   Step 1: G     = A_cm @ A_cm^T = A^T A       [n_order, n_order]
//   Step 2: G_inv = inv(G)                       [n_order, n_order]  (matinvBatched)
//   Step 3: A_pinv = G_inv @ A_cm = inv(A^T A) @ A^T  [n_order, n_freq]
void compute_pseudo_inverse(const core::resources& resources,
                            const core::stream_resources& stream_res, const float* d_A,
                            float* d_Apinv, int n_freq, int n_order)
{
  auto handle = stream_res.cublas_handle;
  const float alpha = 1.0f;
  const float beta = 0.0f;

  float* d_G = resources.alloc_async<float>(n_order * n_order, stream_res);
  float* d_Ginv = resources.alloc_async<float>(n_order * n_order, stream_res);
  int* d_info = resources.alloc_async<int>(1, stream_res);
  float** d_G_ptrs = resources.alloc_async<float*>(1, stream_res);
  float** d_Ginv_ptrs = resources.alloc_async<float*>(1, stream_res);

  // Step 1: G = A^T @ A
  CHECK_CUBLAS(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, n_order, n_order, n_freq, &alpha, d_A,
                           n_order, d_A, n_order, &beta, d_G, n_order));

  // Step 2: G_inv = inv(G)
  // matinvBatched expects device arrays of device pointers;
  // copy host-side pointer values to device so cuBLAS can read them
  CHECK_CUDA(cudaMemcpyAsync(d_G_ptrs, &d_G, sizeof(float*), cudaMemcpyHostToDevice,
                             stream_res.cuda_stream));
  CHECK_CUDA(cudaMemcpyAsync(d_Ginv_ptrs, &d_Ginv, sizeof(float*), cudaMemcpyHostToDevice,
                             stream_res.cuda_stream));
  CHECK_CUBLAS(
      cublasSmatinvBatched(handle, n_order, d_G_ptrs, n_order, d_Ginv_ptrs, n_order, d_info, 1));

  // Step 3: A_pinv = G_inv @ A^T
  CHECK_CUBLAS(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n_order, n_freq, n_order, &alpha,
                           d_Ginv, n_order, d_A, n_order, &beta, d_Apinv, n_order));

  resources.free_async(d_Ginv_ptrs, stream_res);
  resources.free_async(d_G_ptrs, stream_res);
  resources.free_async(d_info, stream_res);
  resources.free_async(d_Ginv, stream_res);
  resources.free_async(d_G, stream_res);
}

// Fused kernel: compute spectral coefficients at peak location
//   Step 1: wy[f] = sqrt_w[f] * dirty[f, peak_row, peak_col]
//   Step 2: compact[o] = sum_f A_pinv(o, f) * wy[f]   (one thread per order, loop over freq)
//   Step 3: per_chan[f] = sum_o A(f, o) * compact[o]    (one thread per freq, loop over order)
//
// A_pinv: col-major [n_order, n_freq]  (from compute_pseudo_inverse)
// A:      row-major [n_freq, n_order]  (from compute_spectral_matrix_kernel)
constexpr int SPECTRAL_BLOCK_SIZE = 128;
__global__ void compute_spectral_coeffs_kernel(float* __restrict__ compact_out,
                                               float* __restrict__ per_chan_out,
                                               const float* __restrict__ dirty,
                                               const float* __restrict__ weights,
                                               const float* __restrict__ A_pinv,
                                               const float* __restrict__ A, int n_freq, int n_order,
                                               int nrow, int ncol, int peak_row, int peak_col)
{
  extern __shared__ float smem[];
  float* s_wy = smem;                // [n_freq]
  float* s_compact = s_wy + n_freq;  // [n_order]

  const int tid = threadIdx.x;
  const int spatial_stride = nrow * ncol;
  const int peak_offset = peak_row * ncol + peak_col;

  // Step 1: build weighted vector at peak pixel
  for (int f = tid; f < n_freq; f += SPECTRAL_BLOCK_SIZE) {
    s_wy[f] = sqrtf(weights[f]) * dirty[f * spatial_stride + peak_offset];
  }
  __syncthreads();

  // Step 2: compact[o] = A_pinv[o, :] . wy
  // A_pinv col-major [n_order, n_freq]: element (o, f) at [f * n_order + o]
  // One thread per order, each loops over all frequencies
  if (tid < n_order) {
    float sum = 0.0f;
    for (int f = 0; f < n_freq; f++) {
      sum += A_pinv[f * n_order + tid] * s_wy[f];
    }
    s_compact[tid] = sum;
    compact_out[tid] = sum;
  }
  __syncthreads();

  // Step 3: per_chan[f] = A[f, :] . compact
  // A row-major [n_freq, n_order]: element (f, o) at [f * n_order + o]
  // One thread per freq, each loops over all orders
  for (int f = tid; f < n_freq; f += SPECTRAL_BLOCK_SIZE) {
    float sum = 0.0f;
    for (int o = 0; o < n_order; o++) {
      sum += A[f * n_order + o] * s_compact[o];
    }
    per_chan_out[f] = sum;
  }
}

void compute_spectral_coeffs(const core::resources& resources,
                             const core::stream_resources& stream_res, float* compact_out,
                             float* per_chan_out, const float* dirty, const float* weights,
                             const float* A_pinv, const float* A, int n_freq, int n_order, int nrow,
                             int ncol, int peak_row, int peak_col)
{
  size_t smem_bytes = (n_freq + n_order) * sizeof(float);

  compute_spectral_coeffs_kernel<<<1, SPECTRAL_BLOCK_SIZE, smem_bytes, stream_res.cuda_stream>>>(
      compact_out, per_chan_out, dirty, weights, A_pinv, A, n_freq, n_order, nrow, ncol, peak_row,
      peak_col);
}

// Subtract PSF from dirty across all frequencies at peak location
// dirty[f, y, x] -= per_chan[f] * gain * psf[f, y, x]  (within overlap region)
__global__ void spectral_psf_subtract_kernel(float* __restrict__ dirty,
                                             const float* __restrict__ psf,
                                             const float* __restrict__ per_chan, overlap_region ovr,
                                             float gain, int n_freq, int dirty_spatial_stride,
                                             int psf_spatial_stride)
{
  const int roi_size = ovr.w * ovr.h;
  const int total = n_freq * roi_size;

  for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < total;
       idx += blockDim.x * gridDim.x) {
    const int f = idx / roi_size;
    const int spatial = idx % roi_size;
    const int ly = spatial / ovr.w;
    const int lx = spatial % ovr.w;

    const int a_offset = f * dirty_spatial_stride + (ovr.ay0 + ly) * ovr.lda + (ovr.ax0 + lx);
    const int b_offset = f * psf_spatial_stride + (ovr.by0 + ly) * ovr.ldb + (ovr.bx0 + lx);

    dirty[a_offset] -= per_chan[f] * gain * psf[b_offset];
  }
}

// Full spectral component subtraction pipeline:
//   1. A = sqrt(jones_norm[f, peak]) * WXdes[f, o]
//   2. A_pinv = inv(A^T A) @ A^T
//   3. wy = sqrt_w * dirty[f, peak]  →  compact = A_pinv @ wy  →  per_chan = A @ compact
//   4. dirty[f] -= per_chan[f] * gain * psf[f]  (within overlap region)
void subtract_component(const core::resources& resources, const core::stream_resources& stream_res,
                        float* dirty, const float* psf, const float* xdes, const float* jones_norm,
                        const float* weights, float gain, int n_freq, int n_order, int nrow,
                        int ncol, int psf_nrow, int psf_ncol, int peak_row, int peak_col)
{
  auto cuda_stream = stream_res.cuda_stream;

  // Allocate temporaries
  float* d_A = resources.alloc_async<float>(n_freq * n_order, stream_res);
  float* d_A_pinv = resources.alloc_async<float>(n_order * n_freq, stream_res);
  float* d_compact = resources.alloc_async<float>(n_order, stream_res);
  float* d_per_chan = resources.alloc_async<float>(n_freq, stream_res);

  // Step 1: build spectral matrix A
  const int n_A = n_freq * n_order;
  compute_spectral_matrix_kernel<<<1, n_A, 0, cuda_stream>>>(
      d_A, xdes, jones_norm, weights, n_freq, n_order, nrow, ncol, peak_row, peak_col);

  // Step 2: pseudo-inverse
  compute_pseudo_inverse(resources, stream_res, d_A, d_A_pinv, n_freq, n_order);

  // Step 3: spectral coefficients
  compute_spectral_coeffs(resources, stream_res, d_compact, d_per_chan, dirty, weights, d_A_pinv,
                          d_A, n_freq, n_order, nrow, ncol, peak_row, peak_col);

  // Step 4: subtract PSF from dirty
  overlap_region ovr = compute_overlap_region(peak_row, peak_col, nrow, ncol, psf_nrow, psf_ncol);
  const int total = n_freq * ovr.w * ovr.h;
  spectral_psf_subtract_kernel<<<CEIL_DIV(total, 256), 256, 0, cuda_stream>>>(
      dirty, psf, d_per_chan, ovr, gain, n_freq, nrow * ncol, psf_nrow * psf_ncol);

  // Free temporaries
  resources.free_async(d_per_chan, stream_res);
  resources.free_async(d_compact, stream_res);
  resources.free_async(d_A_pinv, stream_res);
  resources.free_async(d_A, stream_res);
}

void wscms_minor_cycles_host_loop(const core::resources& resources,
                                  core::device_span4d<float>& residual, float* mean_residual,
                                  const core::device_span6d<float>& psfs,
                                  const core::device_span4d<float>& psfs_2, int scale_idx,
                                  WSCMS_ctx ctx, WSCMS_params params)
{
  const auto& stream_res = resources.get_stream_resources();
  const auto& stream_res_2 = resources.get_stream_resources();
  auto cuda_stream = stream_res.cuda_stream;

  const int nrow = residual.extent(2);
  const int ncol = residual.extent(3);
  const int psf_nrow = psfs_2.extent(2);
  const int psf_ncol = psfs_2.extent(3);
  const int n = nrow * ncol;

  const int n_freq = ctx.xdes.extent(0);
  const int n_order = ctx.xdes.extent(1);

  float* mean_residual_ptr = mean_residual;
  float* residual_ptr = residual.data_handle();
  float* xdes_ptr = ctx.xdes.data_handle();
  float* jones_norm_ptr = ctx.jones_norm.data_handle();
  float* weights_ptr = ctx.weights_freq.data_handle();

  // Allocate output for DeviceReduce::ArgMax
  using KVPair = cub::KeyValuePair<int, float>;
  KVPair* d_argmax_out = resources.alloc_async<KVPair>(1, stream_res);

  // Query and allocate temp storage for DeviceReduce::ArgMax
  size_t temp_storage_bytes = 0;
  cub::DeviceReduce::ArgMax(nullptr, temp_storage_bytes, mean_residual_ptr, d_argmax_out, n,
                            cuda_stream);
  char* d_temp = resources.alloc_async<char>(temp_storage_bytes, stream_res);

  // Initial full argmax
  cub::DeviceReduce::ArgMax(d_temp, temp_storage_bytes, mean_residual_ptr, d_argmax_out, n,
                            cuda_stream);

  KVPair h_peak;
  CHECK_CUDA(
      cudaMemcpyAsync(&h_peak, d_argmax_out, sizeof(KVPair), cudaMemcpyDeviceToHost, cuda_stream));
  stream_res.sync();

  // Threshold is a fraction of the peak
  const float threshold = h_peak.value * params.peak_factor;

  // Mask out below-threshold pixels so they are never modified by PSF subtraction
  apply_threshold_mask_kernel<<<CEIL_DIV(n, 256), 256, 0, cuda_stream>>>(mean_residual_ptr, n,
                                                                         threshold);

  FD_LOG_INFO("minor_loop: initial_peak={:.8f} threshold={:.8f} max_iter={}", h_peak.value,
              threshold, params.max_subminor_iter);
  FD_LOG_DEBUG("minor_loop: grid=[{},{}] psf=[{},{}] n_freq={} n_order={}", nrow, ncol, psf_nrow,
               psf_ncol, n_freq, n_order);

  int n_iter = 0;
  while (h_peak.value > threshold && n_iter < params.max_subminor_iter) {
    auto [peak_row, peak_col] = unravel_index_2D(h_peak.key, ncol);
    int facet_idx = ctx.map_pixel_facet(peak_row, peak_col);
    float gain = ctx.gains(scale_idx, facet_idx);
    float factor = gain * h_peak.value;

    FD_LOG_DEBUG("minor_loop: iter={} peak={:.8f} at ({},{}) facet={} gain={:.4f} factor={}", n_iter,
                 h_peak.value, peak_row, peak_col, facet_idx, gain, factor);

    // Stream 1: PSF subtraction from mean dirty
    const float* psf_2_ptr = psfs_2.data_handle() + psfs_2.mapping()(scale_idx, facet_idx, 0, 0);
    overlap_region ovr = compute_overlap_region(peak_row, peak_col, nrow, ncol, psf_nrow, psf_ncol);
    psf_subtract_kernel<<<CEIL_DIV(ovr.w * ovr.h, 256), 256, 0, cuda_stream>>>(
        mean_residual_ptr, psf_2_ptr, ovr, factor);

    // Stream 2: subtract component from dirty
    const float* psf_ptr = psfs.data_handle() + psfs.mapping()(scale_idx, facet_idx, 0, 0, 0, 0);
    subtract_component(resources, stream_res_2, residual_ptr, psf_ptr, xdes_ptr, jones_norm_ptr,
                       weights_ptr, gain, n_freq, n_order, nrow, ncol, psf_nrow, psf_ncol, peak_row,
                       peak_col);

    // Stream 1: Find peak
    cub::DeviceReduce::ArgMax(d_temp, temp_storage_bytes, mean_residual_ptr, d_argmax_out, n,
                              cuda_stream);
    CHECK_CUDA(cudaMemcpyAsync(&h_peak, d_argmax_out, sizeof(KVPair), cudaMemcpyDeviceToHost,
                               cuda_stream));

    stream_res.sync();
    stream_res_2.sync();

    n_iter++;
  }

  FD_LOG_INFO("minor_loop: finished after {} iterations, final_peak={:.8f}", n_iter, h_peak.value);

  // Cleanup
  resources.free_async(d_temp, stream_res);
  resources.free_async(d_argmax_out, stream_res);
  stream_res.sync();
}

// namespace peak_finding_policy {
// struct naive {
//   static void run() {}
// };
// struct tiled {
//   static void run() {}
// };
// }  // namespace peak_finding_policy
//
// namespace minor_loop_policy {
// template <typename peak_policy>
// struct host_loop {
//   static void run() { peak_policy::run(); }
// };
// struct kernel_loop {
//   static void run() {}
// };
// }  // namespace minor_loop_policy
//
// template <typename minorloop_policy>
// void run_wscms()
// {
//   // find best kernel
//
//   minorloop_policy::run();
// }
//
// void bar() { run_wscms<minor_loop_policy::host_loop<peak_finding_policy::naive>>(); }

}  // namespace fast_deconv::algorithm::wscms::detail
