#pragma once
#include <array>
#include <cufft.h>
#include <fast_deconv/util/cuda_macros.hpp>

#ifndef CUFFT_CALL
#define CUFFT_CALL(call)                                                       \
  {                                                                            \
    auto status = static_cast<cufftResult>(call);                              \
    if (status != CUFFT_SUCCESS)                                               \
      fprintf(stderr,                                                          \
              "ERROR: CUFFT call \"%s\" in line %d of file %s failed "         \
              "with "                                                          \
              "code (%d).\n",                                                  \
              #call, __LINE__, __FILE__, status);                              \
  }
#endif // CUFFT_CALL

using complex_type = cufftComplex;

namespace fast_deconv::algorithm::wscms::detail {

// Multiplies freq_dirty with each scale and normalizes by 1/N
// freq_total = dirty_x * (dirty_y / 2 + 1) (half-complex from R2C)
__global__ void multiply_batched_kernel(complex_type *freq_dirty,
                                        complex_type *scales,
                                        complex_type *scaled_dirty,
                                        int freq_total, int n_scales,
                                        float norm) {
  uint tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= freq_total)
    return;

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
void scale_convolve(float *dirty, complex_type *scales, float *out_scaled_dirty,
                    int dirty_x, int dirty_y, int scale_x, int scale_y,
                    int n_scales) {
  int freq_total = scale_x * scale_y; // dirty_x * (dirty_y / 2 + 1)

  cufftHandle plan_forward, plan_backward;
  CUFFT_CALL(cufftPlan2d(&plan_forward, dirty_x, dirty_y, CUFFT_R2C));
  std::array<int, 2> fft_size{dirty_x, dirty_y};
  CUFFT_CALL(cufftPlanMany(&plan_backward, fft_size.size(), fft_size.data(),
                           nullptr, 1, 0, // *inembed, istride, idist
                           nullptr, 1, 0, // *onembed, ostride, odist
                           CUFFT_C2R, n_scales));

  cudaStream_t stream = NULL;
  CHECK_CUDA(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  CUFFT_CALL(cufftSetStream(plan_forward, stream));
  CUFFT_CALL(cufftSetStream(plan_backward, stream));

  complex_type *dirty_freq = nullptr;
  cudaMallocAsync(reinterpret_cast<void **>(&dirty_freq),
                  sizeof(complex_type) * freq_total, stream);
  complex_type *scaled_dirty_freq = nullptr;
  cudaMallocAsync(reinterpret_cast<void **>(&scaled_dirty_freq),
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

} // namespace fast_deconv::algorithm::wscms::detail
