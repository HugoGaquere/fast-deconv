#pragma once

// The template body below launches a kernel, so every source that includes this must be compiled by nvcc.
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

}  // namespace fast_deconv::kernel

namespace fast_deconv::core {

/// f(i) for every i in [0, n), one thread each, enqueued on @p ctx. @p f must be FD_HOST_DEVICE.
template <typename F>
void for_each_index(const exec_ctx& ctx, std::int64_t n, F f)
{
  launch(ctx, launch_cfg(n), kernel::for_each_index_kernel<F>, n, f);
}

}  // namespace fast_deconv::core
