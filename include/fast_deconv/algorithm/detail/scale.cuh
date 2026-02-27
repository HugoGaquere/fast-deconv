#pragma once
#include <array>
#include <fast_deconv/linalg/detail/fft.cuh>

namespace fast_deconv::algorithm::wscms::detail {

#define PI 3.141592654f
#define PI_SQUARRED 9.869604403f

__global__ void make_scales_kernel(float* sigmas, int scale_x, int scale_y, int n_scales,
                                   float* scales)
{
  const int row = blockIdx.y * blockDim.y + threadIdx.y;
  const int col = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= scale_x || col >= scale_y) return;

  const float freq_x = row < (scale_x + 1) / 2 ? static_cast<float>(row) / scale_x
                                               : static_cast<float>(row - scale_x) / scale_x;
  const float freq_y = col < (scale_y + 1) / 2 ? static_cast<float>(col) / scale_y
                                               : static_cast<float>(col - scale_y) / scale_y;
  const float rhosq = freq_x * freq_x + freq_y * freq_y;

  const int scale_size = scale_x * scale_y;
  const int tid = row * scale_y + col;

  for (int i = 0; i < n_scales; i++) {
    const uint idx = tid + i * scale_size;
    const float sigma = sigmas[i];
    scales[idx] = exp(-2.0f * PI_SQUARRED * rhosq * sigma * sigma);
  }
}

__global__ void make_scales_kernel_half(float* sigmas, int scale_x, int scale_y_half,
                                        int scale_y_full, int n_scales, float* scales)
{
  // col only ranges from 0 to ny/2 (ny/2+1 values)
  // All frequencies are non-negative: no conditional branch needed

  const int row = blockIdx.y * blockDim.y + threadIdx.y;
  const int col = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= scale_x || col >= scale_y_half) return;

  const float freq_x = row < (scale_x + 1) / 2 ? static_cast<float>(row) / scale_x
                                               : static_cast<float>(row - scale_x) / scale_x;
  const float freq_y = static_cast<float>(col) / scale_y_full;

  const float rhosq = freq_x * freq_x + freq_y * freq_y;

  const int scale_size = scale_x * scale_y_half;
  const int tid = row * scale_y_half + col;

  for (int i = 0; i < n_scales; i++) {
    const uint idx = tid + i * scale_size;
    const float sigma = sigmas[i];
    scales[idx] = exp(-2.0f * PI_SQUARRED * rhosq * sigma * sigma);
  }
}

void make_scales(float* sigmas, int scale_x, int scale_y_half, int scale_y_full, int n_scales,
                 float* scales, cudaStream_t stream)
{
  dim3 block_dim(16, 16);
  dim3 grid_dim(CEIL_DIV(scale_y_half, block_dim.x), CEIL_DIV(scale_x, block_dim.y));
  make_scales_kernel_half<<<grid_dim, block_dim, 0, stream>>>(sigmas, scale_x, scale_y_half,
                                                              scale_y_full, n_scales, scales);
}

// Multiplies freq_dirty with each scale and normalizes by 1/N
// freq_total = dirty_x * (dirty_y / 2 + 1) (half-complex from R2C)
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

// Convolves a dirty image with Gaussian scale kernels.
// dirty:            (npix_x, npix_y) real, device
// sigmas:           (n_scales,) Gaussian sigmas, device
// out_scaled_dirty: (n_scales, npix_x, npix_y) real, device
// Internally: pad+ifftshift → R2C → multiply with Gaussian scales → C2R → fftshift+crop
void scale_convolve(float* dirty, float* sigmas, float* out_scaled_dirty, int npix_x, int npix_y,
                    int n_scales, float padding)
{
  const auto [npad_x, npad_y] = linalg::detail::compute_padding(npix_x, npix_y, padding);
  const int npadded_x = npix_x + 2 * npad_x;
  const int npadded_y = npix_y + 2 * npad_y;
  const int freq_x = npadded_x;
  const int freq_y = npadded_y / 2 + 1;
  const int freq_total = freq_x * freq_y;

  // Create cuFFT plans (synchronous, allocates workspace on default stream)
  cufftHandle plan_forward, plan_backward;
  CUFFT_CALL(cufftPlan2d(&plan_forward, npadded_x, npadded_y, CUFFT_R2C));
  std::array<int, 2> fft_size{npadded_x, npadded_y};
  CUFFT_CALL(cufftPlanMany(&plan_backward, 2, fft_size.data(), nullptr, 1, 0, nullptr, 1, 0,
                           CUFFT_C2R, n_scales));

  // Allocate temporaries
  cudaStream_t stream = NULL;
  CHECK_CUDA(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  CUFFT_CALL(cufftSetStream(plan_forward, stream));
  CUFFT_CALL(cufftSetStream(plan_backward, stream));

  float* dirty_padded = nullptr;
  float* scales = nullptr;
  complex_type* dirty_freq = nullptr;
  complex_type* scaled_dirty_freq = nullptr;
  float* scaled_dirty = nullptr;
  CHECK_CUDA(cudaMallocAsync(reinterpret_cast<void**>(&dirty_padded),
                             sizeof(float) * npadded_x * npadded_y, stream));
  CHECK_CUDA(cudaMallocAsync(reinterpret_cast<void**>(&scales),
                             sizeof(float) * freq_total * n_scales, stream));
  CHECK_CUDA(cudaMallocAsync(reinterpret_cast<void**>(&dirty_freq),
                             sizeof(complex_type) * freq_total, stream));
  CHECK_CUDA(cudaMallocAsync(reinterpret_cast<void**>(&scaled_dirty_freq),
                             sizeof(complex_type) * freq_total * n_scales, stream));
  CHECK_CUDA(cudaMallocAsync(reinterpret_cast<void**>(&scaled_dirty),
                             sizeof(float) * npadded_x * npadded_y * n_scales, stream));

  // Pad + ifftshift dirty image
  linalg::detail::pad_ifftshift(dirty, dirty_padded, npix_x, npix_y, npadded_x, npadded_y, npad_x,
                                npad_y, stream);

  // Generate Gaussian scale kernels in half-complex frequency domain
  make_scales(sigmas, npadded_x, freq_y, npadded_y, n_scales, scales, stream);

  // Forward R2C: real (npadded_x, npadded_y) -> half-complex (npadded_x, npadded_y/2+1)
  CUFFT_CALL(cufftExecR2C(plan_forward, dirty_padded, dirty_freq));

  // Element-wise multiply with scales + 1/N normalization
  float norm = 1.0f / static_cast<float>(npadded_x * npadded_y);
  multiply_batched_kernel<<<CEIL_DIV(freq_total, 256), 256, 0, stream>>>(
      dirty_freq, scales, scaled_dirty_freq, freq_total, n_scales, norm);

  // Inverse C2R: half-complex -> real per batch
  CUFFT_CALL(cufftExecC2R(plan_backward, scaled_dirty_freq, scaled_dirty));

  // Fftshift + crop back to original size
  linalg::detail::fftshift_crop(scaled_dirty, out_scaled_dirty, npix_x, npix_y, npadded_x,
                                npadded_y, npad_x, npad_y, n_scales, stream);

  // Cleanup
  CHECK_CUDA(cudaFreeAsync(dirty_padded, stream));
  CHECK_CUDA(cudaFreeAsync(scales, stream));
  CHECK_CUDA(cudaFreeAsync(dirty_freq, stream));
  CHECK_CUDA(cudaFreeAsync(scaled_dirty_freq, stream));
  CHECK_CUDA(cudaFreeAsync(scaled_dirty, stream));
  CUFFT_CALL(cufftDestroy(plan_forward));
  CUFFT_CALL(cufftDestroy(plan_backward));
  CHECK_CUDA(cudaStreamSynchronize(stream));
  CHECK_CUDA(cudaStreamDestroy(stream));
}

}  // namespace fast_deconv::algorithm::wscms::detail
