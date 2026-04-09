#pragma once
#include <cufft.h>

#include <algorithm>
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
                                  int mask_stride, bool clean_negative)
{
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= npix) return;

  for (int s = 0; s < n_scales; s++) {
    const int idx = s * npix + tid;
    const bool masked = mask[s * mask_stride + tid];
    if (masked) {
      scaled_dirty[idx] = -INFINITY;
    } else if (clean_negative) {
      scaled_dirty[idx] = fabsf(scaled_dirty[idx]);
    }
  }
}

}  // namespace

namespace fast_deconv::algorithm::wscms::detail {

void make_scales(const core::resources& resources, const core::stream_resources& stream_res,
                 float* sigmas, int scale_nrow, int scale_ncol_half, int scale_ncol_full,
                 int n_scales, float* scales)
{
  dim3 block_dim(16, 16);
  dim3 grid_dim(CEIL_DIV(scale_ncol_half, block_dim.x), CEIL_DIV(scale_nrow, block_dim.y));
  make_scales_kernel_half<<<grid_dim, block_dim, 0, stream_res.cuda_stream>>>(
      sigmas, scale_nrow, scale_ncol_half, scale_ncol_full, n_scales, scales);
  stream_res.sync();
}

// Finds the best scale and peak pixel via biased peak-finding.
// scaled_dirty is modified in-place (mask + abs applied).
// mask: device, True = masked. Either (npix,) shared or (n_scales, npix) per-scale.
// bias: (n_scales,) host memory, multiplied with per-scale peaks to select best scale.
// per_scale_mask: false = shared mask (npix,), true = per-scale masks (n_scales, npix)
// retired_scales: scale indices to exclude from selection.
// Returns unbiased peak value and pixel coordinates.
scale_selection_result scale_selection(const core::resources& resources,
                                       const core::stream_resources& stream_res,
                                       float* scaled_dirty, const bool* mask, const float* bias,
                                       int n_scales, int nrow, int ncol, bool clean_negative,
                                       bool per_scale_mask,
                                       const std::vector<int>& retired_scales = {})
{
  const auto cuda_stream = stream_res.cuda_stream;

  const int npix = nrow * ncol;
  const int mask_stride = per_scale_mask ? npix : 0;

  // 1. Apply mask + abs in-place
  apply_mask_kernel<<<CEIL_DIV(npix, 256), 256, 0, cuda_stream>>>(scaled_dirty, mask, npix,
                                                                  n_scales, mask_stride, clean_negative);

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

  // 5. Biased scale selection on host (skip retired scales)
  int best_scale = -1;
  float best_biased = -INFINITY;
  for (int s = 0; s < n_scales; s++) {
    if (std::find(retired_scales.begin(), retired_scales.end(), s) != retired_scales.end()) continue;
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

scale_convole_ctx make_scale_convolve_ctx(int nrow, int ncol, int n_scales, float padding)
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
void scale_convolve(const core::resources& resources, const core::stream_resources& stream_res,
                    const scale_convole_ctx& ctx, float* dirty, float* scales,
                    float* out_scaled_dirty, int n_scales)
{
  const int img_padded_total = ctx.img_padded_nrow * ctx.img_padded_ncol;
  const int freq_total = ctx.freq_nrow * ctx.freq_ncol;
  const int freq_scales_total = freq_total * n_scales;

  cudaStream_t cuda_stream = stream_res.cuda_stream;
  CUFFT_CALL(cufftSetStream(ctx.plan_forward, cuda_stream));
  CUFFT_CALL(cufftSetStream(ctx.plan_backward, cuda_stream));

  // Allocate temporaries
  float* dirty_padded = resources.alloc_async<float>(img_padded_total, stream_res);
  complex_type* dirty_freq = resources.alloc_async<complex_type>(freq_total, stream_res);
  complex_type* scaled_dirty_freq =
      resources.alloc_async<complex_type>(freq_scales_total, stream_res);
  float* scaled_dirty = resources.alloc_async<float>(img_padded_total * n_scales, stream_res);

  stream_res.sync();

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

  // we want untouched mean dirty at scale 0
  cudaMemcpyAsync(out_scaled_dirty, dirty, ctx.img_nrow * ctx.img_ncol * sizeof(float),
                  cudaMemcpyDeviceToDevice, stream_res.cuda_stream);

  // Cleanup
  resources.free_async(dirty_padded, stream_res);
  resources.free_async(dirty_freq, stream_res);
  resources.free_async(scaled_dirty_freq, stream_res);
  resources.free_async(scaled_dirty, stream_res);

  stream_res.sync();
}

/**
 * @brief Copies the selected scale slice from a batched buffer to a flat output.
 *
 * Extracts the slice at index @p best_scale from @p scaled_dirty laid out as
 * (n_scales, npix) and copies it into @p out (npix).
 *
 * @param stream_res    CUDA stream resources for the async copy.
 * @param scaled_dirty  Device pointer to the batched buffer, shape (n_scales, npix).
 * @param out           Device pointer to the output buffer, size npix.
 * @param best_scale    Index of the scale slice to extract.
 * @param npix          Number of pixels per scale (nrow * ncol).
 */
void copy_scale_slice(const core::stream_resources& stream_res, const float* scaled_dirty,
                      float* out, int best_scale, int npix)
{
  const float* src = scaled_dirty + best_scale * npix;
  cudaMemcpyAsync(out, src, npix * sizeof(float), cudaMemcpyDeviceToDevice, stream_res.cuda_stream);
}

// ================================================================== //
//     PSF convolution for a single scale
// ================================================================== //

/**
 * @brief   Multiply batched freq-domain PSFs by a single Gaussian kernel,
 *          producing both G and G^2 variants.
 * @details For each frequency pixel and each batch, computes:
 *          freq_conv[b,i]  = freq_psf[b,i] * scale[i] * norm
 *          freq_conv2[b,i] = freq_psf[b,i] * scale[i]^2 * norm
 *
 * @param[in]  freq_psf     Input half-complex PSFs, (n_batch, freq_total).
 * @param[in]  scale_kernel Gaussian kernel in freq domain, size freq_total.
 * @param[out] freq_conv    Output for single-convolved PSFs.
 * @param[out] freq_conv2   Output for double-convolved PSFs.
 * @param[in]  freq_total   Number of complex elements per slice (freq_nrow * freq_ncol).
 * @param[in]  n_batch      Number of 2D slices.
 * @param[in]  norm         Normalization factor (1 / padded_total).
 */
__global__ void multiply_psf_scale_kernel(const complex_type* freq_psf, const float* scale_kernel,
                                          complex_type* freq_conv, complex_type* freq_conv2,
                                          int freq_total, int n_batch, float norm)
{
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= freq_total) return;

  const float g = scale_kernel[tid];
  const float g_norm = g * norm;
  const float g2_norm = g * g * norm;

  for (int b = 0; b < n_batch; b++) {
    const int idx = b * freq_total + tid;
    const complex_type val = freq_psf[idx];
    freq_conv[idx] = {val.x * g_norm, val.y * g_norm};
    freq_conv2[idx] = {val.x * g2_norm, val.y * g2_norm};
  }
}

/**
 * @brief   Compute weighted mean over channels for each spatial pixel.
 * @details Computes output[i] = sum_c( input[c * npix + i] * weights[c] )
 *          for each pixel i in [0, npix).
 *
 * @param[in]  input   Channel images, layout (nch, npix), device.
 * @param[in]  weights Per-channel weights, size nch, device.
 * @param[out] output  Weighted mean image, size npix, device.
 * @param[in]  npix    Number of spatial pixels.
 * @param[in]  nch     Number of channels.
 */
__global__ void weighted_mean_channels_kernel(const float* __restrict__ input,
                                              const float* __restrict__ weights,
                                              float* __restrict__ output, int npix, int nch)
{
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < npix; i += blockDim.x * gridDim.x) {
    float sum = 0.0f;
    for (int c = 0; c < nch; c++) {
      sum += input[c * npix + i] * weights[c];
    }
    output[i] = sum;
  }
}

/**
 * @brief   Create a PSF convolution context with FFT plans for batched PSF processing.
 *
 * @param[in] psf_nrow Number of rows in the PSF.
 * @param[in] psf_ncol Number of columns in the PSF.
 * @param[in] nch      Batch size (number of frequency channels).
 * @param[in] padding  FFT padding factor (e.g. 1.5).
 *
 * @return psf_convolve_ctx with initialized FFT plans.
 */
psf_convolve_ctx make_psf_convolve_ctx(int psf_nrow, int psf_ncol, int nch, float padding)
{
  const auto [npad_row, npad_col] = linalg::detail::compute_padding(psf_nrow, psf_ncol, padding);

  psf_convolve_ctx ctx;
  ctx.psf_nrow = psf_nrow;
  ctx.psf_ncol = psf_ncol;
  ctx.padding_nrow = npad_row;
  ctx.padding_ncol = npad_col;
  ctx.psf_padded_nrow = psf_nrow + 2 * npad_row;
  ctx.psf_padded_ncol = psf_ncol + 2 * npad_col;
  ctx.freq_nrow = ctx.psf_padded_nrow;
  ctx.freq_ncol = ctx.psf_padded_ncol / 2 + 1;
  ctx.n_batch = nch;

  std::array<int, 2> fft_size{ctx.psf_padded_nrow, ctx.psf_padded_ncol};
  CUFFT_CALL(cufftPlanMany(&ctx.plan_forward, 2, fft_size.data(), nullptr, 1, 0, nullptr, 1, 0,
                           CUFFT_R2C, nch));
  CUFFT_CALL(cufftPlanMany(&ctx.plan_backward, 2, fft_size.data(), nullptr, 1, 0, nullptr, 1, 0,
                           CUFFT_C2R, nch));
  CUFFT_CALL(cufftPlanMany(&ctx.plan_backward_2, 2, fft_size.data(), nullptr, 1, 0, nullptr, 1, 0,
                           CUFFT_C2R, nch));
  return ctx;
}

/**
 * @brief   Convolve raw PSFs with Gaussian(sigma) for all facets, producing
 *          single-convolved and double-convolved (weighted mean) PSFs.
 * @details For each facet, batches over nch frequency channels:
 *          - conv_psf  = PSF * G(sigma)      [per-channel]
 *          - conv2_mean = wmean(PSF * G^2)    [weighted mean over channels]
 *
 *          Scale 0 (sigma == 0) is handled as a fast path: conv_psf is a
 *          device-to-device copy and conv2_mean is a weighted channel mean.
 *
 * @param[in]  resources    GPU memory allocator.
 * @param[in]  stream_res   CUDA stream resources.
 * @param[in]  ctx          PSF convolution context (FFT plans, padding).
 * @param[in]  raw_psfs     Raw PSFs, device, layout (n_facets, nch, npol, psf_h, psf_w).
 * @param[in]  sigma        Gaussian sigma for this scale.
 * @param[in]  weights      Per-channel weights, device, size nch.
 * @param[in]  n_facets     Number of facets.
 * @param[in]  nch          Number of frequency channels.
 * @param[out] out_conv_psf  Output single-convolved PSFs, device,
 *                           layout (n_facets, nch, 1, psf_h, psf_w), pre-allocated.
 * @param[out] out_conv2_mean Output double-convolved weighted mean PSFs, device,
 *                            layout (n_facets, psf_h, psf_w), pre-allocated.
 */
void convolve_psfs_for_scale(const core::resources& resources,
                             const core::stream_resources& stream_res,
                             const psf_convolve_ctx& ctx, const float* raw_psfs, float sigma,
                             const float* weights, int n_facets, int nch, float* out_conv_psf,
                             float* out_conv2_mean)
{
  const auto cuda_stream = stream_res.cuda_stream;
  const int psf_npix = ctx.psf_nrow * ctx.psf_ncol;
  const int facet_stride = nch * psf_npix;  // npol=1 after squeeze
  const int padded_total = ctx.psf_padded_nrow * ctx.psf_padded_ncol;
  const int freq_total = ctx.freq_nrow * ctx.freq_ncol;

  // Scale 0 fast path: no convolution needed
  if (sigma == 0.0f) {
    cudaMemcpyAsync(out_conv_psf, raw_psfs, sizeof(float) * n_facets * facet_stride,
                    cudaMemcpyDeviceToDevice, cuda_stream);
    for (int f = 0; f < n_facets; f++) {
      weighted_mean_channels_kernel<<<CEIL_DIV(psf_npix, 256), 256, 0, cuda_stream>>>(
          raw_psfs + f * facet_stride, weights, out_conv2_mean + f * psf_npix, psf_npix, nch);
    }
    stream_res.sync();
    return;
  }

  CUFFT_CALL(cufftSetStream(ctx.plan_forward, cuda_stream));
  CUFFT_CALL(cufftSetStream(ctx.plan_backward, cuda_stream));
  CUFFT_CALL(cufftSetStream(ctx.plan_backward_2, cuda_stream));

  // Allocate temporaries
  float* padded_psf = resources.alloc_async<float>(nch * padded_total, stream_res);
  complex_type* freq_psf = resources.alloc_async<complex_type>(nch * freq_total, stream_res);
  complex_type* freq_conv = resources.alloc_async<complex_type>(nch * freq_total, stream_res);
  complex_type* freq_conv2 = resources.alloc_async<complex_type>(nch * freq_total, stream_res);
  float* padded_conv = resources.alloc_async<float>(nch * padded_total, stream_res);
  float* padded_conv2 = resources.alloc_async<float>(nch * padded_total, stream_res);
  float* conv2_cropped = resources.alloc_async<float>(nch * psf_npix, stream_res);

  // Generate Gaussian scale kernel at PSF resolution (single kernel, reused for all facets)
  float* scale_kernel = resources.alloc_async<float>(freq_total, stream_res);
  float sigma_arr = sigma;
  float* d_sigma = resources.alloc_async<float>(1, stream_res);
  CHECK_CUDA(
      cudaMemcpyAsync(d_sigma, &sigma_arr, sizeof(float), cudaMemcpyHostToDevice, cuda_stream));
  make_scales_kernel_half<<<dim3(CEIL_DIV(ctx.freq_ncol, 16), CEIL_DIV(ctx.freq_nrow, 16)),
                            dim3(16, 16), 0, cuda_stream>>>(
      d_sigma, ctx.freq_nrow, ctx.freq_ncol, ctx.psf_padded_ncol, 1, scale_kernel);
  resources.free_async(d_sigma, stream_res);

  const float norm = 1.0f / static_cast<float>(padded_total);

  for (int f = 0; f < n_facets; f++) {
    const float* src = raw_psfs + f * facet_stride;
    float* dst_conv = out_conv_psf + f * facet_stride;
    float* dst_conv2_mean = out_conv2_mean + f * psf_npix;

    // 1. Pad + ifftshift (batched over nch)
    linalg::detail::pad_ifftshift_batched(const_cast<float*>(src), padded_psf, ctx.psf_nrow,
                                          ctx.psf_ncol, ctx.psf_padded_nrow, ctx.psf_padded_ncol,
                                          ctx.padding_nrow, ctx.padding_ncol, nch, cuda_stream);

    // 2. Batched R2C FFT
    CUFFT_CALL(cufftExecR2C(ctx.plan_forward, padded_psf, freq_psf));

    // 3. Multiply by G and G^2
    multiply_psf_scale_kernel<<<CEIL_DIV(freq_total, 256), 256, 0, cuda_stream>>>(
        freq_psf, scale_kernel, freq_conv, freq_conv2, freq_total, nch, norm);

    // 4. Batched C2R IFFT for conv_psf
    CUFFT_CALL(cufftExecC2R(ctx.plan_backward, freq_conv, padded_conv));

    // 5. Batched C2R IFFT for conv2_psf
    CUFFT_CALL(cufftExecC2R(ctx.plan_backward_2, freq_conv2, padded_conv2));

    // 6. fftshift + crop for conv_psf -> output
    linalg::detail::fftshift_crop(padded_conv, dst_conv, ctx.psf_nrow, ctx.psf_ncol,
                                  ctx.psf_padded_nrow, ctx.psf_padded_ncol, ctx.padding_nrow,
                                  ctx.padding_ncol, nch, cuda_stream);

    // 7. fftshift + crop for conv2_psf -> temporary
    linalg::detail::fftshift_crop(padded_conv2, conv2_cropped, ctx.psf_nrow, ctx.psf_ncol,
                                  ctx.psf_padded_nrow, ctx.psf_padded_ncol, ctx.padding_nrow,
                                  ctx.padding_ncol, nch, cuda_stream);

    // 8. Weighted mean over channels -> conv2_mean output
    weighted_mean_channels_kernel<<<CEIL_DIV(psf_npix, 256), 256, 0, cuda_stream>>>(
        conv2_cropped, weights, dst_conv2_mean, psf_npix, nch);
  }

  // Cleanup temporaries
  resources.free_async(conv2_cropped, stream_res);
  resources.free_async(padded_conv2, stream_res);
  resources.free_async(padded_conv, stream_res);
  resources.free_async(freq_conv2, stream_res);
  resources.free_async(freq_conv, stream_res);
  resources.free_async(freq_psf, stream_res);
  resources.free_async(padded_psf, stream_res);
  resources.free_async(scale_kernel, stream_res);

  stream_res.sync();
}

}  // namespace fast_deconv::algorithm::wscms::detail
