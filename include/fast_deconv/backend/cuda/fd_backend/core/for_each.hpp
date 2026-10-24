#pragma once

// The template bodies below launch kernels, so every source that includes this must be compiled by nvcc.
#ifndef __CUDACC__
#error "fd_backend/core/for_each.hpp needs nvcc: add the including source to FAST_DECONV_GENERIC_SOURCES"
#endif

#include <cstdint>
#include <fast_deconv/core/exec_ctx.hpp>
#include <fd_backend/core/launch.cuh>

namespace fast_deconv::kernel {

template <typename F>
__global__ void for_each_index_kernel(std::int64_t n, F f)
{
  const std::int64_t i = std::int64_t{blockIdx.x} * blockDim.x + threadIdx.x;
  if (i < n) f(i);
}

template <typename F>
__global__ void for_each_index_2d_kernel(std::int64_t nrow, std::int64_t ncol, F f)
{
  const std::int64_t r = std::int64_t{blockIdx.y} * blockDim.y + threadIdx.y;
  const std::int64_t c = std::int64_t{blockIdx.x} * blockDim.x + threadIdx.x;
  if (r < nrow && c < ncol) f(r, c);
}

template <typename F>
__global__ void for_each_index_3d_kernel(std::int64_t nrow, std::int64_t ncol, F f)
{
  const std::int64_t r = std::int64_t{blockIdx.y} * blockDim.y + threadIdx.y;
  const std::int64_t c = std::int64_t{blockIdx.x} * blockDim.x + threadIdx.x;
  if (r < nrow && c < ncol) f(std::int64_t{blockIdx.z}, r, c);
}

}  // namespace fast_deconv::kernel

namespace fast_deconv::core {

/// f(i) for every i in [0, n), one thread each, enqueued on @p ctx. @p f must be FD_HOST_DEVICE.
template <typename F>
void for_each_index(const exec_ctx& ctx, std::int64_t n, F f)
{
  launch(ctx, launch_cfg(n), kernel::for_each_index_kernel<F>, n, f);
}

/// f(r, c) over an @p nrow x @p ncol plane, one thread per pixel, a warp along a row.
template <typename F>
void for_each_index(const exec_ctx& ctx, std::int64_t nrow, std::int64_t ncol, F f)
{
  launch(ctx, launch_cfg(nrow, ncol), kernel::for_each_index_2d_kernel<F>, nrow, ncol, f);
}

/// f(d, r, c) over @p depth planes of @p nrow x @p ncol, one grid z-plane each; @p depth at most 65535.
template <typename F>
void for_each_index(const exec_ctx& ctx, std::int64_t depth, std::int64_t nrow, std::int64_t ncol, F f)
{
  launch(ctx, launch_cfg(depth, nrow, ncol), kernel::for_each_index_3d_kernel<F>, nrow, ncol, f);
}

}  // namespace fast_deconv::core
