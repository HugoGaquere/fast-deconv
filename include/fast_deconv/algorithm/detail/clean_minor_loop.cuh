#pragma once
#include <cooperative_groups.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cfloat>
#include <cub/cub.cuh>
#include <fast_deconv/algorithm/wscms_types.hpp>
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

// Compute aligned bounding boxes to extract overlapping sub-regions from
// arrays A and B, where B is centered at a given position within A.
// lda and ldb are computed as the array widths.
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

// psfs_2 layout: (n_scale, n_facet, psf_nrow, psf_ncol)
void wscms_minor_cycles_host_loop(const core::resources& resources,
                                  core::device_span2d<float>& residual,
                                  core::device_span4d<float>& psfs_2,
                                  core::host_span2d<int>& map_pixels_facets,
                                  core::host_span2d<float>& gains, int scale_idx, float threshold,
                                  int max_iter)
{
  const auto& stream_res = resources.get_stream_resources();
  auto cuda_stream = stream_res.cuda_stream;

  const int nrow = residual.extent(0);
  const int ncol = residual.extent(1);
  const int n_facet = psfs_2.extent(1);
  const int psf_nrow = psfs_2.extent(2);
  const int psf_ncol = psfs_2.extent(3);
  const int n = nrow * ncol;

  float* d_residual = residual.data_handle();
  float* d_psfs = psfs_2.data_handle() + scale_idx * n_facet * psf_nrow * psf_ncol;

  // Allocate output for DeviceReduce::ArgMax
  using KVPair = cub::KeyValuePair<int, float>;
  KVPair* d_argmax_out = resources.alloc_async<KVPair>(1, stream_res);

  // Query and allocate temp storage for DeviceReduce::ArgMax
  size_t temp_storage_bytes = 0;
  cub::DeviceReduce::ArgMax(nullptr, temp_storage_bytes, d_residual, d_argmax_out, n, cuda_stream);
  char* d_temp = resources.alloc_async<char>(temp_storage_bytes, stream_res);

  // Initial full argmax
  cub::DeviceReduce::ArgMax(d_temp, temp_storage_bytes, d_residual, d_argmax_out, n, cuda_stream);

  KVPair h_peak;
  CHECK_CUDA(
      cudaMemcpyAsync(&h_peak, d_argmax_out, sizeof(KVPair), cudaMemcpyDeviceToHost, cuda_stream));
  stream_res.sync();

  int n_iter = 0;
  while (h_peak.value > threshold && n_iter < max_iter) {
    auto [peak_row, peak_col] = unravel_index_2D(h_peak.key, ncol);
    int facet_id = map_pixels_facets(peak_row, peak_col);
    float gain = gains(scale_idx, facet_id);
    float factor = gain * h_peak.value;

    const float* psf_facet = d_psfs + facet_id * psf_nrow * psf_ncol;
    overlap_region ovr = compute_overlap_region(peak_row, peak_col, nrow, ncol, psf_nrow, psf_ncol);

    // PSF subtraction
    int roi_size = ovr.w * ovr.h;
    psf_subtract_kernel<<<CEIL_DIV(roi_size, 256), 256, 0, cuda_stream>>>(d_residual, psf_facet,
                                                                          ovr, factor);

    // Find peak
    cub::DeviceReduce::ArgMax(d_temp, temp_storage_bytes, d_residual, d_argmax_out, n, cuda_stream);

    CHECK_CUDA(cudaMemcpyAsync(&h_peak, d_argmax_out, sizeof(KVPair), cudaMemcpyDeviceToHost,
                               cuda_stream));
    stream_res.sync();

    n_iter++;
  }

  resources.free_async(d_temp, stream_res);
  resources.free_async(d_argmax_out, stream_res);
}

}  // namespace fast_deconv::algorithm::wscms::detail
