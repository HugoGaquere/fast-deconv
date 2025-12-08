#pragma once

#include <fast_deconv/core/stream_resources.hpp>
#include <fast_deconv/util/cuda_macros.hpp>

namespace fast_deconv::matrix::detail {

__global__ void subtract_kernel(const float* A, const float* B, float* C, size_t size)
{
  const uint tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < size) { C[tid] = A[tid] - B[tid]; }
}

void subtract_async(
  const float* A, const float* B, float* C, size_t size, core::stream_resources& resources)
{
  int block_size = 256;
  int grid_size  = CEIL_DIV(size, static_cast<size_t>(block_size));
  auto stream    = resources.stream;

  subtract_kernel<<<grid_size, block_size, 0, stream>>>(A, B, C, size);
  CHECK_LAST_CUDA_ERROR();
}

}  // namespace fast_deconv::matrix::detail
