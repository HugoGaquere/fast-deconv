#pragma once
#include <cufft.h>

#include <array>
#include <fast_deconv/util/cuda_macros.hpp>

#ifndef CUFFT_CALL
#define CUFFT_CALL(call)                                               \
  {                                                                    \
    auto status = static_cast<cufftResult>(call);                      \
    if (status != CUFFT_SUCCESS)                                       \
      fprintf(stderr,                                                  \
              "ERROR: CUFFT call \"%s\" in line %d of file %s failed " \
              "with "                                                  \
              "code (%d).\n",                                          \
              #call, __LINE__, __FILE__, status);                      \
  }
#endif  // CUFFT_CALL

using complex_type = cufftComplex;

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
                 float* scales)
{
  cudaStream_t stream = NULL;
  CHECK_CUDA(cudaStreamCreate(&stream));
  dim3 block_dim(16, 16);
  dim3 grid_dim(CEIL_DIV(scale_y_half, block_dim.x), CEIL_DIV(scale_x, block_dim.y));
  make_scales_kernel_half<<<grid_dim, block_dim, 0, stream>>>(sigmas, scale_x, scale_y_half,
                                                              scale_y_full, n_scales, scales);
  CHECK_CUDA(cudaStreamSynchronize(stream));
  CHECK_CUDA(cudaStreamDestroy(stream));
}

// Multiplies freq_dirty with each scale and normalizes by 1/N
// freq_total = dirty_x * (dirty_y / 2 + 1) (half-complex from R2C)
__global__ void multiply_batched_kernel(complex_type* freq_dirty, complex_type* scales,
                                        complex_type* scaled_dirty, int freq_total, int n_scales,
                                        float norm)
{
  uint tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= freq_total) return;

  const complex_type dirty_val = freq_dirty[tid];

  for (int i = 0; i < n_scales; i++) {
    const int idx = tid + i * freq_total;
    complex_type result = cuCmulf(dirty_val, scales[idx]);
    result.x *= norm;
    result.y *= norm;
    scaled_dirty[idx] = result;
  }
}

// dirty: (dirty_x, dirty_y) real
// scales: (n_scales, scale_x, scale_y) half-complex from rfft2
//         where scale_x = dirty_x, scale_y = dirty_y / 2 + 1
// out_scaled_dirty: (n_scales, dirty_x, dirty_y) real
void scale_convolve(float* dirty, complex_type* scales, float* out_scaled_dirty, int dirty_x,
                    int dirty_y, int scale_x, int scale_y, int n_scales)
{
  int freq_total = scale_x * scale_y;  // dirty_x * (dirty_y / 2 + 1)

  cufftHandle plan_forward, plan_backward;
  CUFFT_CALL(cufftPlan2d(&plan_forward, dirty_x, dirty_y, CUFFT_R2C));
  std::array<int, 2> fft_size{dirty_x, dirty_y};
  CUFFT_CALL(cufftPlanMany(&plan_backward, fft_size.size(), fft_size.data(), nullptr, 1,
                           0,              // *inembed, istride, idist
                           nullptr, 1, 0,  // *onembed, ostride, odist
                           CUFFT_C2R, n_scales));

  cudaStream_t stream = NULL;
  CHECK_CUDA(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  CUFFT_CALL(cufftSetStream(plan_forward, stream));
  CUFFT_CALL(cufftSetStream(plan_backward, stream));

  complex_type* dirty_freq = nullptr;
  cudaMallocAsync(reinterpret_cast<void**>(&dirty_freq), sizeof(complex_type) * freq_total, stream);
  complex_type* scaled_dirty_freq = nullptr;
  cudaMallocAsync(reinterpret_cast<void**>(&scaled_dirty_freq),
                  sizeof(complex_type) * freq_total * n_scales, stream);

  // Forward R2C: real (dirty_x, dirty_y) → half-complex (dirty_x, dirty_y/2+1)
  CUFFT_CALL(cufftExecR2C(plan_forward, dirty, dirty_freq));

  // Element-wise multiply with 1/N normalization
  float norm = 1.0f / static_cast<float>(dirty_x * dirty_y);
  multiply_batched_kernel<<<CEIL_DIV(freq_total, 256), 256, 0, stream>>>(
      dirty_freq, scales, scaled_dirty_freq, freq_total, n_scales, norm);

  // Inverse C2R: half-complex → real (dirty_x, dirty_y) per batch
  CUFFT_CALL(cufftExecC2R(plan_backward, scaled_dirty_freq, out_scaled_dirty));

  CHECK_CUDA(cudaFreeAsync(dirty_freq, stream));
  CHECK_CUDA(cudaFreeAsync(scaled_dirty_freq, stream));
  CUFFT_CALL(cufftDestroy(plan_forward));
  CUFFT_CALL(cufftDestroy(plan_backward));
  CHECK_CUDA(cudaStreamDestroy(stream));
}

}  // namespace fast_deconv::algorithm::wscms::detail
