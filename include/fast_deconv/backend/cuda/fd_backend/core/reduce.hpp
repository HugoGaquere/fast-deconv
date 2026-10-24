#pragma once

// The template body below instantiates CUB kernels, so every source that includes this must be compiled by nvcc.
#ifndef __CUDACC__
#error "fd_backend/core/reduce.hpp needs nvcc: add the including source to FAST_DECONV_GENERIC_SOURCES"
#endif

#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cub/device/device_reduce.cuh>
#include <fast_deconv/core/exec_ctx.hpp>
#include <fast_deconv/util/cuda_macros.hpp>

namespace fast_deconv::core {

/**
 * @brief combine over map(i), i in [0, n), returned on host. Syncs @p ctx.
 *
 * @p map and @p combine must be FD_HOST_DEVICE; @p combine associative and commutative, @p identity its identity.
 */
template <typename T, typename Map, typename Combine>
T reduce(const exec_ctx& ctx, std::int64_t n, Map map, Combine combine, T identity)
{
  const auto it = thrust::make_transform_iterator(thrust::counting_iterator<std::int64_t>{0}, map);
  auto d_result = ctx.alloc_ptr<T>(1);

  std::size_t temp_bytes = 0;
  CHECK_CUDA(cub::DeviceReduce::Reduce(nullptr, temp_bytes, it, d_result.get(), n, combine, identity, ctx.cuda_stream));
  auto d_temp = ctx.alloc_ptr<std::byte>(narrow<std::int64_t>(std::max<std::size_t>(temp_bytes, 1)));
  CHECK_CUDA(
      cub::DeviceReduce::Reduce(d_temp.get(), temp_bytes, it, d_result.get(), n, combine, identity, ctx.cuda_stream));

  T result{};
  ctx.copy(&result, d_result.get(), 1);
  ctx.wait();
  return result;
}

}  // namespace fast_deconv::core
