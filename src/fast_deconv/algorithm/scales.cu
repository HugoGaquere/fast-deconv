#include <algorithm>
#include <array>
#include <cstdint>
#include <cub/cub.cuh>
#include <emu/submdspan.hpp>
#include <fast_deconv/algorithm/scales.hpp>
#include <fast_deconv/core/logger.hpp>
#include <fast_deconv/linalg/fft.hpp>
#include <fast_deconv/linalg/linalg.hpp>
#include <stdexcept>

namespace fast_deconv::kernel {
#define PI 3.141592654f
#define PI_SQUARRED 9.869604403f

__global__ void make_scales_kernel_half(const float* sigmas, int scale_nrow, int scale_ncol_half, int scale_ncol_full,
                                        int n_scales, float* scales)
{
  const int row = blockIdx.y * blockDim.y + threadIdx.y;
  const int col = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= scale_nrow || col >= scale_ncol_half) return;

  const float freq_row = row < (scale_nrow + 1) / 2 ? static_cast<float>(row) / scale_nrow
                                                    : static_cast<float>(row - scale_nrow) / scale_nrow;
  const float freq_col = static_cast<float>(col) / scale_ncol_full;

  const float rhosq = freq_row * freq_row + freq_col * freq_col;

  const int scale_size = scale_nrow * scale_ncol_half;
  const int tid = row * scale_ncol_half + col;

  float* out = scales + tid;
  for (int i = 0; i < n_scales; i++, out += scale_size) {
    const float sigma = sigmas[i];
    *out = exp(-2.0f * PI_SQUARRED * rhosq * sigma * sigma);
  }
}

// Multiplies freq_dirty with each scale and normalizes by 1/N
// freq_total = nrow * (ncol / 2 + 1) (half-complex from R2C)
// Grid: (freq_total / blockDim.x, n_scales). One thread per (freq bin, scale).
__global__ void multiply_batched_kernel(complex_type* freq_dirty, float* scales, complex_type* scaled_dirty,
                                        int freq_total, int n_scales, float norm)
{
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  const int scale_id = blockIdx.y;
  if (tid >= freq_total || scale_id >= n_scales) return;

  const complex_type dirty_val = freq_dirty[tid];
  const std::int64_t idx = tid + static_cast<std::int64_t>(scale_id) * freq_total;
  const float scale_norm = scales[idx] * norm;
  scaled_dirty[idx] = {dirty_val.x * scale_norm, dirty_val.y * scale_norm};
}

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
                                          complex_type* freq_conv, complex_type* freq_conv2, int freq_total,
                                          int n_batch, float norm)
{
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= freq_total) return;

  const float g = scale_kernel[tid];
  const float g_norm = g * norm;
  const float g2_norm = g * g * norm;

  const complex_type* in = freq_psf + tid;
  complex_type* out_conv = freq_conv + tid;
  complex_type* out_conv2 = freq_conv2 + tid;
  for (int b = 0; b < n_batch; b++, in += freq_total, out_conv += freq_total, out_conv2 += freq_total) {
    const complex_type val = *in;
    *out_conv = {val.x * g_norm, val.y * g_norm};
    *out_conv2 = {val.x * g2_norm, val.y * g2_norm};
  }
}

}  // namespace fast_deconv::kernel

namespace fast_deconv::scale {
void make_gaussian_kernels_async(const core::stream_resources& stream_res, core::device_vect<float> sigmas,
                                 int scale_ncol_full, core::device_span3d<float> scales)
{
  const int n_scales = scales.extent(0);
  const int scale_nrow = scales.extent(1);
  const int scale_ncol_half = scales.extent(2);
  dim3 block_dim(16, 16);
  dim3 grid_dim(CEIL_DIV(scale_ncol_half, block_dim.x), CEIL_DIV(scale_nrow, block_dim.y));
  kernel::make_scales_kernel_half<<<grid_dim, block_dim, 0, stream_res.cuda_stream>>>(
      sigmas.data_handle(), scale_nrow, scale_ncol_half, scale_ncol_full, n_scales, scales.data_handle());
}

void convolve_with_scales(const algorithm::ddmsc::scale_convolve_ctx& ctx, core::device_span2d<float> dirty,
                          core::device_span3d<float> scales, core::device_span3d<float> out_scaled_dirty)
{
  const core::stream_resources& stream_res = ctx.stream_res;  // plans run on this stream
  const int n_scales = out_scaled_dirty.extent(0);
  const int img_padded_total = ctx.padded_nrow * ctx.padded_ncol;
  const int freq_total = ctx.freq_nrow * ctx.freq_ncol;
  const int npix = ctx.input_nrow * ctx.input_ncol;

  cudaStream_t cuda_stream = stream_res.cuda_stream;

  // Scale 0 (sigma == 0) is the identity kernel — copy the input directly.
  CHECK_CUDA(cudaMemcpyAsync(out_scaled_dirty.data_handle(), dirty.data_handle(), npix * sizeof(float),
                             cudaMemcpyDeviceToDevice, cuda_stream));

  if (n_scales <= 1) return;

  // Plan was built with this batch size; loop over chunks of that size to cover
  // all (n_scales - 1) non-trivial scales. (n_scales - 1) must be a multiple of
  // backward_batch — the cuFFT C2R plan always processes exactly batch slices,
  // so a remainder would OOB-read scales[] and OOB-write out_scaled_dirty[].
  const int chunk_batch = ctx.backward_batch;
  const int total_scales = n_scales - 1;
  if (total_scales % chunk_batch != 0) {
    throw std::invalid_argument("convolve_with_scales: (n_scales - 1) must be a multiple of backward_batch_size");
  }

  // Plans were bound to their stream at construction (the ddmsc context's
  // compute stream); temporaries are stream-ordered on that same stream.
  auto dirty_padded = stream_res.alloc_mdcontainer_async<float>(img_padded_total);
  auto dirty_freq = stream_res.alloc_mdcontainer_async<complex_type>(freq_total);
  auto scaled_dirty_freq = stream_res.alloc_mdcontainer_async<complex_type>(chunk_batch, freq_total);
  auto scaled_dirty = stream_res.alloc_mdcontainer_async<float>(chunk_batch, img_padded_total);

  // Pad + ifftshift dirty image
  linalg::pad_ifftshift(dirty.data_handle(), dirty_padded.data_handle(), ctx.input_nrow, ctx.input_ncol,
                        ctx.padded_nrow, ctx.padded_ncol, ctx.padding_nrow, ctx.padding_ncol, cuda_stream);

  // Forward R2C — once, reused across all chunks
  CUFFT_CALL(cufftExecR2C(ctx.plan_forward(), dirty_padded.data_handle(), dirty_freq.data_handle()));

  // Loop over chunks of chunk_batch scales (skip scale 0, identity, already memcpy'd above).
  const float norm = 1.0f / static_cast<float>(img_padded_total);
  for (int chunk_off = 0; chunk_off < total_scales; chunk_off += chunk_batch) {
    const int scale_idx_start = 1 + chunk_off;

    dim3 mul_grid(CEIL_DIV(freq_total, 256), chunk_batch);
    kernel::multiply_batched_kernel<<<mul_grid, 256, 0, cuda_stream>>>(
        dirty_freq.data_handle(), scales.data_handle() + static_cast<std::int64_t>(scale_idx_start) * freq_total,
        scaled_dirty_freq.data_handle(), freq_total, chunk_batch, norm);

    CUFFT_CALL(cufftExecC2R(ctx.plan_backward(), scaled_dirty_freq.data_handle(), scaled_dirty.data_handle()));

    linalg::fftshift_crop(scaled_dirty.data_handle(),
                          out_scaled_dirty.data_handle() + static_cast<std::int64_t>(scale_idx_start) * npix,
                          ctx.input_nrow, ctx.input_ncol, ctx.padded_nrow, ctx.padded_ncol, ctx.padding_nrow,
                          ctx.padding_ncol, chunk_batch, cuda_stream);
  }
}

int scale_selection(const core::stream_resources& stream_res, core::device_span3d<float> scaled_dirty,
                    core::host_vect<float> bias, const std::vector<int>& retired_scales)
{
  const auto cuda_stream = stream_res.cuda_stream;

  const int n_scales = scaled_dirty.extent(0);
  const int nrow = scaled_dirty.extent(1);
  const int ncol = scaled_dirty.extent(2);
  const int npix = nrow * ncol;

  // Per-scale peak via a loop of full-image Max reductions. CUB's
  // DeviceSegmentedReduce parallelizes ACROSS segments, so with only n_scales
  // (~5) huge segments it leaves the GPU almost idle. A DeviceReduce::Max per
  // scale saturates the GPU on every segment instead. We only need the peak
  // VALUE per scale (the argmax index is never used downstream), so Max — not
  // ArgMax — suffices. Masked-out pixels are already -inf, so they never win.
  auto d_maxes = stream_res.alloc_mdcontainer_async<float>(n_scales);

  size_t temp_bytes = 0;
  CHECK_CUDA(cub::DeviceReduce::Max(nullptr, temp_bytes, scaled_dirty.data_handle(), d_maxes.data_handle(), npix,
                                    cuda_stream));
  auto d_temp = stream_res.alloc_mdcontainer_async<char>(temp_bytes);

  for (int s = 0; s < n_scales; s++) {
    CHECK_CUDA(cub::DeviceReduce::Max(d_temp.data_handle(), temp_bytes,
                                      scaled_dirty.data_handle() + static_cast<std::int64_t>(s) * npix,
                                      d_maxes.data_handle() + s, npix, cuda_stream));
  }

  // Copy per-scale peaks to host
  std::vector<float> h_maxes(n_scales);
  CHECK_CUDA(cudaMemcpyAsync(h_maxes.data(), d_maxes.data_handle(), sizeof(float) * n_scales, cudaMemcpyDeviceToHost,
                             cuda_stream));

  stream_res.sync();

  // Biased scale selection on host (skip retired scales)
  int best_scale = 0;
  float best_biased = -std::numeric_limits<float>::infinity();
  for (int s = 0; s < n_scales; s++) {
    if (std::find(retired_scales.begin(), retired_scales.end(), s) != retired_scales.end()) continue;
    float biased = h_maxes[s] * bias[s];
    if (biased > best_biased) {
      best_biased = biased;
      best_scale = s;
    }
  }

  return best_scale;
}

void convolve_psfs_with_scale_async(const algorithm::ddmsc::psf_convolve_ctx& ctx, core::device_span4d<float> psfs,
                                    core::device_vect<float> d_sigma, int scale_idx, core::device_vect<float> weights,
                                    core::device_span4d<float> out_conv_psf, core::device_span3d<float> out_conv2_mean)
{
  const core::stream_resources& stream_res = ctx.stream_res;  // plans run on this stream
  const int n_facets = psfs.extent(0);
  const int n_freq = psfs.extent(1);
  const int psf_npix = ctx.input_nrow * ctx.input_ncol;
  const int facet_stride = n_freq * psf_npix;
  const int padded_total = ctx.padded_nrow * ctx.padded_ncol;
  const int freq_total = ctx.freq_nrow * ctx.freq_ncol;

  // Scale 0 fast path: no convolution needed
  if (scale_idx == 0) {
    CHECK_CUDA(cudaMemcpyAsync(out_conv_psf.data_handle(), psfs.data_handle(), sizeof(float) * n_facets * facet_stride,
                               cudaMemcpyDeviceToDevice, stream_res.cuda_stream));
    for (int f = 0; f < n_facets; f++) {
      linalg::weighted_sum_async(stream_res, core::device_span3d<float>(emu::submdspan(psfs, f)), weights,
                                 core::device_span2d<float>(emu::submdspan(out_conv2_mean, f)));
    }
    return;
  }

  const auto cuda_stream = stream_res.cuda_stream;

  // Allocate temporaries
  auto padded_psf = stream_res.alloc_mdcontainer_async<float>(n_freq, padded_total);
  auto freq_psf = stream_res.alloc_mdcontainer_async<complex_type>(n_freq, freq_total);
  auto freq_conv = stream_res.alloc_mdcontainer_async<complex_type>(n_freq, freq_total);
  auto freq_conv2 = stream_res.alloc_mdcontainer_async<complex_type>(n_freq, freq_total);
  auto padded_conv = stream_res.alloc_mdcontainer_async<float>(n_freq, padded_total);
  auto padded_conv2 = stream_res.alloc_mdcontainer_async<float>(n_freq, padded_total);
  auto conv2_cropped = stream_res.alloc_mdcontainer_async<float>(n_freq, ctx.input_nrow, ctx.input_ncol);

  // Generate Gaussian scale kernel at PSF resolution (single kernel, reused for all facets)
  auto scale_kernel = stream_res.alloc_mdcontainer_async<float>(1, ctx.freq_nrow, ctx.freq_ncol);
  make_gaussian_kernels_async(stream_res, d_sigma, ctx.padded_ncol, scale_kernel);

  const float norm = 1.0f / static_cast<float>(padded_total);

  // Advanced per facet: n_facets * facet_stride exceeds int32 at production sizes
  float* src = psfs.data_handle();
  float* dst_conv = out_conv_psf.data_handle();
  float* dst_conv2_mean = out_conv2_mean.data_handle();

  for (int f = 0; f < n_facets; f++) {
    // 1. Pad + ifftshift (batched over n_freq)
    linalg::pad_ifftshift_batched(src, padded_psf.data_handle(), ctx.input_nrow, ctx.input_ncol, ctx.padded_nrow,
                                  ctx.padded_ncol, ctx.padding_nrow, ctx.padding_ncol, n_freq, cuda_stream);

    // 2. Batched R2C FFT
    CUFFT_CALL(cufftExecR2C(ctx.plan_forward(), padded_psf.data_handle(), freq_psf.data_handle()));

    // 3. Multiply by G and G^2
    kernel::multiply_psf_scale_kernel<<<CEIL_DIV(freq_total, 256), 256, 0, cuda_stream>>>(
        freq_psf.data_handle(), scale_kernel.data_handle(), freq_conv.data_handle(), freq_conv2.data_handle(),
        freq_total, n_freq, norm);

    // 4. Batched C2R IFFT for conv_psf
    CUFFT_CALL(cufftExecC2R(ctx.plan_backward(), freq_conv.data_handle(), padded_conv.data_handle()));

    // 5. Batched C2R IFFT for conv2_psf
    CUFFT_CALL(cufftExecC2R(ctx.plan_backward_2(), freq_conv2.data_handle(), padded_conv2.data_handle()));

    // 6. fftshift + crop for conv_psf -> output
    linalg::fftshift_crop(padded_conv.data_handle(), dst_conv, ctx.input_nrow, ctx.input_ncol, ctx.padded_nrow,
                          ctx.padded_ncol, ctx.padding_nrow, ctx.padding_ncol, n_freq, cuda_stream);

    // 7. fftshift + crop for conv2_psf -> temporary
    linalg::fftshift_crop(padded_conv2.data_handle(), conv2_cropped.data_handle(), ctx.input_nrow, ctx.input_ncol,
                          ctx.padded_nrow, ctx.padded_ncol, ctx.padding_nrow, ctx.padding_ncol, n_freq, cuda_stream);

    // 8. Weighted mean over channels -> conv2_mean output
    core::device_span2d<float> dst_conv2_mean_view(dst_conv2_mean, ctx.input_nrow, ctx.input_ncol);
    linalg::weighted_sum_async(stream_res, conv2_cropped, weights, dst_conv2_mean_view);

    src += facet_stride;
    dst_conv += facet_stride;
    dst_conv2_mean += psf_npix;
  }
}

void convolve_psfs_with_scales_async(const algorithm::ddmsc::psf_convolve_ctx& ctx, core::device_span4d<float> psfs,
                                     core::device_vect<float> d_sigmas, core::device_vect<float> weights,
                                     core::device_span5d<float> out_conv_psf, core::device_span4d<float> out_conv2_mean)
{
  const int n_scales = d_sigmas.size();
  for (int i = 0; i < n_scales; i++) {
    core::device_vect<float> sigma_view(d_sigmas.data_handle() + i, 1);
    core::device_span4d<float> current_conv_psf = emu::submdspan(out_conv_psf, i);
    core::device_span3d<float> current_conv2_psf = emu::submdspan(out_conv2_mean, i);
    convolve_psfs_with_scale_async(ctx, psfs, sigma_view, i, weights, current_conv_psf, current_conv2_psf);
  }
}

}  // namespace fast_deconv::scale
