#pragma once

#include <cub/cub.cuh>

#include <fast_deconv/core/stream_resources.hpp>
#include <fast_deconv/util/cuda_macros.hpp>

namespace fast_deconv::matrix::detail {

// template <typename Mdspan>
// __global__ void simple_subtract_kernel(const Mdspan A, const Mdspan B, Mdspan C)
// {
//   const uint tix = blockIdx.x * blockDim.x + threadIdx.x;
//
//   constexpr uint size = A.size();
//   if (tid >= size) return;
//
//   if constexpr()
// }

__global__ void subtract_kernel_vect_load(const float* __restrict__ A,
                                          const float* __restrict__ B,
                                          float* __restrict__ C,
                                          size_t size)
{
  const size_t tid    = blockIdx.x * blockDim.x + threadIdx.x;
  const size_t stride = blockDim.x * gridDim.x;

  const auto* a = reinterpret_cast<const float4*>(A);
  const auto* b = reinterpret_cast<const float4*>(B);
  auto* c       = reinterpret_cast<float4*>(C);

  for (int i = tid; i < size / 4; i += stride) {
    float4 va = a[i];
    float4 vb = b[i];
    float4 vc;
    vc.x = va.x - vb.x;
    vc.y = va.y - vb.y;
    vc.z = va.z - vb.z;
    vc.w = va.w - vb.w;

    c[i] = vc;
  }

  // in only one thread, process final elements (if there are any)
  int remainder = size % 4;
  if (tid == 0 && remainder != 0) {
    while (remainder) {
      int idx = size - remainder--;
      C[idx]  = A[idx] - B[idx];
    }
  }
}

template <typename T,
          int BLOCK_THREADS,
          int ITEMS_PER_THREAD,
          cub::BlockLoadAlgorithm LOAD_ALGORITHM,
          cub::BlockStoreAlgorithm STORE_ALGORITHM>
__global__ void subtract_kernel_cub_load(const T* __restrict__ A,
                                         const T* __restrict__ B,
                                         T* __restrict__ C,
                                         size_t size)
{
  using block_load  = cub::BlockLoad<T, BLOCK_THREADS, ITEMS_PER_THREAD, LOAD_ALGORITHM>;
  using block_store = cub::BlockStore<T, BLOCK_THREADS, ITEMS_PER_THREAD, STORE_ALGORITHM>;

  // Obtain this block's segment of consecutive items
  T thread_a[ITEMS_PER_THREAD];
  T thread_b[ITEMS_PER_THREAD];
  T thread_c[ITEMS_PER_THREAD];
  int block_offset = blockIdx.x * (BLOCK_THREADS * ITEMS_PER_THREAD);

  block_load().Load(A + block_offset, thread_a);
  block_load().Load(B + block_offset, thread_b);

// Do the subtraction
#pragma unroll ITEMS_PER_THREAD
  for (int i = 0; i < ITEMS_PER_THREAD; ++i) {
    thread_c[i] = thread_a[i] - thread_b[i];
  }

  // Store back to C
  block_store().Store(C + block_offset, thread_c);
}

void subtract_async(
  const float* A, const float* B, float* C, size_t size, core::stream_resources& resources)
{
  auto stream = resources.stream;

  // const int TILE_SIZE = BLOCK_THREADS * ITEMS_PER_THREAD;
  // int grid_size       = static_cast<int>((size + TILE_SIZE - 1) / TILE_SIZE);
  // dim3 block(BLOCK_THREADS);
  // dim3 grid(grid_size);
  // subtract_kernel_cub_load<<<grid, block, 0, stream>>>(A, B, C, size);
  // resources.sync();

  constexpr int items_per_thread = 4;
  constexpr int block_dim        = 128;
  const int grid_dim             = CEIL_DIV(size / items_per_thread, block_dim);

  constexpr auto subtract_kernel = subtract_kernel_cub_load<float,
                                                            block_dim,
                                                            items_per_thread,
                                                            cub::BLOCK_LOAD_VECTORIZE,
                                                            cub::BLOCK_STORE_VECTORIZE>;

  subtract_kernel<<<grid_dim, block_dim, 0, stream>>>(A, B, C, size);
  CHECK_LAST_CUDA_ERROR();
}

}  // namespace fast_deconv::matrix::detail
