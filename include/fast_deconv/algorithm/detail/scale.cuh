#pragma once
#include <cufft.h>

#include <array>
#include <cub/cub.cuh>
#include <fast_deconv/linalg/detail/fft.cuh>
#include <vector>

#include "fast_deconv/algorithm/wscms_types.hpp"
#include "fast_deconv/core/resources.hpp"

namespace {

#define PI 3.141592654f
#define PI_SQUARRED 9.869604403f

__global__ void make_scales_kernel_half(float* sigmas, int scale_nrow, int scale_ncol_half,
                                        int scale_ncol_full, int n_scales, float* scales)
{
  // col only ranges from 0 to ncol/2 (ncol/2+1 values)
  // All frequencies are non-negative: no conditional branch needed

  const int row = blockIdx.y * blockDim.y + threadIdx.y;
  const int col = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= scale_nrow || col >= scale_ncol_half) return;

  const float freq_row = row < (scale_nrow + 1) / 2
                             ? static_cast<float>(row) / scale_nrow
                             : static_cast<float>(row - scale_nrow) / scale_nrow;
  const float freq_col = static_cast<float>(col) / scale_ncol_full;

  const float rhosq = freq_row * freq_row + freq_col * freq_col;

  const int scale_size = scale_nrow * scale_ncol_half;
  const int tid = row * scale_ncol_half + col;

  for (int i = 0; i < n_scales; i++) {
    const uint idx = tid + i * scale_size;
    const float sigma = sigmas[i];
    scales[idx] = exp(-2.0f * PI_SQUARRED * rhosq * sigma * sigma);
  }
}

// Multiplies freq_dirty with each scale and normalizes by 1/N
// freq_total = nrow * (ncol / 2 + 1) (half-complex from R2C)
__global__ void multiply_batched_kernel(complex_type* freq_dirty, float* scales,
                                        complex_type* scaled_dirty, int freq_total, int n_scales,
                                        float norm)
{
  uint tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= freq_total) return;

  const complex_type dirty_val = freq_dirty[tid];

  for (int i = 0; i < n_scales; i++) {
    const int idx = tid + i * freq_total;
    const float scale_norm = scales[idx] * norm;
    scaled_dirty[idx] = {dirty_val.x * scale_norm, dirty_val.y * scale_norm};
  }
}

// Applies mask and optional abs to scaled_dirty in-place.
// scaled_dirty: (n_scales, npix) modified in-place
// mask:         layout depends on mask_stride:
//               mask_stride=0:    (npix,)           shared mask across all scales
//               mask_stride=npix: (n_scales, npix)  per-scale masks
// True = masked (set to -inf)
__global__ void apply_mask_kernel(float* scaled_dirty, const bool* mask, int npix, int n_scales,
                                  int mask_stride, bool do_abs)
{
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= npix) return;

  for (int s = 0; s < n_scales; s++) {
    const int idx = s * npix + tid;
    const bool masked = mask[s * mask_stride + tid];
    if (masked) {
      scaled_dirty[idx] = -INFINITY;
    } else if (do_abs) {
      scaled_dirty[idx] = fabsf(scaled_dirty[idx]);
    }
  }
}

}  // namespace

namespace fast_deconv::algorithm::wscms::detail {

void make_scales(const core::resources& resources, float* sigmas, int scale_nrow,
                 int scale_ncol_half, int scale_ncol_full, int n_scales, float* scales)
{
  const auto& stream_res = resources.get_stream_resources();
  dim3 block_dim(16, 16);
  dim3 grid_dim(CEIL_DIV(scale_ncol_half, block_dim.x), CEIL_DIV(scale_nrow, block_dim.y));
  make_scales_kernel_half<<<grid_dim, block_dim, 0, stream_res.cuda_stream>>>(
      sigmas, scale_nrow, scale_ncol_half, scale_ncol_full, n_scales, scales);
}

// Finds the best scale and peak pixel via biased peak-finding.
// scaled_dirty is modified in-place (mask + abs applied).
// mask: device, True = masked. Either (npix,) shared or (n_scales, npix) per-scale.
// bias: (n_scales,) host memory, multiplied with per-scale peaks to select best scale.
// per_scale_mask: false = shared mask (npix,), true = per-scale masks (n_scales, npix)
// Returns unbiased peak value and pixel coordinates.
scale_selection_result scale_selection(const core::resources& resources, float* scaled_dirty,
                                     const bool* mask, const float* bias, int n_scales, int nrow,
                                     int ncol, bool do_abs, bool per_scale_mask)
{
  const auto& stream_res = resources.get_stream_resources();
  auto cuda_stream = stream_res.cuda_stream;

  const int npix = nrow * ncol;
  const int mask_stride = per_scale_mask ? npix : 0;

  // 1. Apply mask + abs in-place
  apply_mask_kernel<<<CEIL_DIV(npix, 256), 256, 0, cuda_stream>>>(scaled_dirty, mask, npix,
                                                                  n_scales, mask_stride, do_abs);

  // 2. Build segment offsets [0, npix, 2*npix, ..., n_scales*npix]
  int* d_offsets = resources.alloc_async<int>(n_scales + 1, stream_res);
  std::vector<int> h_offsets(n_scales + 1);
  for (int i = 0; i <= n_scales; i++) h_offsets[i] = i * npix;
  CHECK_CUDA(cudaMemcpyAsync(d_offsets, h_offsets.data(), sizeof(int) * (n_scales + 1),
                             cudaMemcpyHostToDevice, cuda_stream));

  // 3. CUB segmented argmax
  using KVPair = cub::KeyValuePair<int, float>;
  KVPair* d_peaks = resources.alloc_async<KVPair>(n_scales, stream_res);

  size_t temp_bytes = 0;
  CHECK_CUDA(cub::DeviceSegmentedReduce::ArgMax(nullptr, temp_bytes, scaled_dirty, d_peaks,
                                                n_scales, d_offsets, d_offsets + 1, cuda_stream));

  void* d_temp = resources.alloc_async(temp_bytes, stream_res);
  CHECK_CUDA(cub::DeviceSegmentedReduce::ArgMax(d_temp, temp_bytes, scaled_dirty, d_peaks, n_scales,
                                                d_offsets, d_offsets + 1, cuda_stream));

  // 4. Copy per-scale peaks to host
  std::vector<KVPair> h_peaks(n_scales);
  CHECK_CUDA(cudaMemcpyAsync(h_peaks.data(), d_peaks, sizeof(KVPair) * n_scales,
                             cudaMemcpyDeviceToHost, cuda_stream));

  stream_res.sync();

  // Async cleanup
  resources.free_async(d_offsets, stream_res);
  resources.free_async(d_peaks, stream_res);
  resources.free_async(d_temp, stream_res);

  // 5. Biased scale selection on host
  int best_scale = 0;
  float best_biased = -INFINITY;
  for (int s = 0; s < n_scales; s++) {
    float biased = h_peaks[s].value * bias[s];
    if (biased > best_biased) {
      best_biased = biased;
      best_scale = s;
    }
  }

  int best_flat_idx = h_peaks[best_scale].key;
  int best_row = best_flat_idx / ncol;
  int best_col = best_flat_idx % ncol;

  return {best_scale, best_row, best_col, h_peaks[best_scale].value};
}

scale_convole_ctx make_scale_convole_ctx(int nrow, int ncol, int n_scales, float padding)
{
  const auto [npad_row, npad_col] = linalg::detail::compute_padding(nrow, ncol, padding);

  scale_convole_ctx ctx;
  ctx.img_nrow = nrow;
  ctx.img_ncol = ncol;
  ctx.padding_nrow = npad_row;
  ctx.padding_ncol = npad_col;
  ctx.img_padded_nrow = nrow + 2 * npad_row;
  ctx.img_padded_ncol = ncol + 2 * npad_col;
  ctx.freq_nrow = ctx.img_padded_nrow;
  ctx.freq_ncol = ctx.img_padded_ncol / 2 + 1;
  ctx.n_batches = n_scales;

  CUFFT_CALL(cufftPlan2d(&ctx.plan_forward, ctx.img_padded_nrow, ctx.img_padded_ncol, CUFFT_R2C));

  std::array<int, 2> fft_size{ctx.img_padded_nrow, ctx.img_padded_ncol};
  CUFFT_CALL(cufftPlanMany(&ctx.plan_backward, 2, fft_size.data(), nullptr, 1, 0, nullptr, 1, 0,
                           CUFFT_C2R, n_scales));

  return ctx;
}

// Convolves a dirty image with Gaussian scale kernels.
// dirty:            (nrow, ncol) real, device
// sigmas:           (n_scales,) Gaussian sigmas, device
// out_scaled_dirty: (n_scales, nrow, ncol) real, device
// Internally: pad+ifftshift → R2C → multiply with Gaussian scales → C2R → fftshift+crop
void scale_convolve(const core::resources& resources, const scale_convole_ctx& ctx, float* dirty,
                    float* sigmas, float* out_scaled_dirty, int n_scales)
{
  const int img_padded_total = ctx.img_padded_nrow * ctx.img_padded_ncol;
  const int freq_total = ctx.freq_nrow * ctx.freq_ncol;
  const int freq_scales_total = freq_total * n_scales;

  const auto& stream_res = resources.get_stream_resources();
  cudaStream_t cuda_stream = stream_res.cuda_stream;
  CUFFT_CALL(cufftSetStream(ctx.plan_forward, cuda_stream));
  CUFFT_CALL(cufftSetStream(ctx.plan_backward, cuda_stream));

  // Allocate temporaries
  float* dirty_padded = resources.alloc_async<float>(img_padded_total, stream_res);
  float* scales = resources.alloc_async<float>(freq_scales_total, stream_res);
  complex_type* dirty_freq = resources.alloc_async<complex_type>(freq_total, stream_res);
  complex_type* scaled_dirty_freq =
      resources.alloc_async<complex_type>(freq_scales_total, stream_res);
  float* scaled_dirty = resources.alloc_async<float>(img_padded_total * n_scales, stream_res);

  stream_res.sync();

  // Generate Gaussian scale kernels in half-complex frequency domain
  make_scales(resources, sigmas, ctx.freq_nrow, ctx.freq_ncol, ctx.img_padded_ncol, n_scales,
              scales, cuda_stream);

  // Pad + ifftshift dirty image
  linalg::detail::pad_ifftshift(dirty, dirty_padded, ctx.img_nrow, ctx.img_ncol,
                                ctx.img_padded_nrow, ctx.img_padded_ncol, ctx.padding_nrow,
                                ctx.padding_ncol, cuda_stream);

  // Forward R2C: real (nrow, ncol) -> half-complex (nrow, ncol/2+1)
  CUFFT_CALL(cufftExecR2C(ctx.plan_forward, dirty_padded, dirty_freq));

  // Element-wise multiply with scales + 1/N normalization
  float norm = 1.0f / static_cast<float>(img_padded_total);
  multiply_batched_kernel<<<CEIL_DIV(freq_total, 256), 256, 0, cuda_stream>>>(
      dirty_freq, scales, scaled_dirty_freq, freq_total, n_scales, norm);

  // Inverse C2R: half-complex -> real per batch
  CUFFT_CALL(cufftExecC2R(ctx.plan_backward, scaled_dirty_freq, scaled_dirty));

  // Fftshift + crop back to original size
  linalg::detail::fftshift_crop(scaled_dirty, out_scaled_dirty, ctx.img_nrow, ctx.img_ncol,
                                ctx.img_padded_nrow, ctx.img_padded_ncol, ctx.padding_nrow,
                                ctx.padding_ncol, n_scales, cuda_stream);

  // Cleanup
  resources.free_async(dirty_padded, stream_res);
  resources.free_async(scales, stream_res);
  resources.free_async(dirty_freq, stream_res);
  resources.free_async(scaled_dirty_freq, stream_res);
  resources.free_async(scaled_dirty, stream_res);

  stream_res.sync();
}

}  // namespace fast_deconv::algorithm::wscms::detail
