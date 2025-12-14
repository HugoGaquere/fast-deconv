#pragma once

#include <fast_deconv/core/stream_resources.hpp>
#include <fast_deconv/matrix/detail/subtract.cuh>

namespace fast_deconv::matrix {

inline void subtract_async(
  const float* A, const float* B, float* C, size_t size, core::stream_resources& resources)
{
  detail::subtract_async(A, B, C, size, resources);
}

inline void subtract(
  const float* A, const float* B, float* C, size_t size, core::stream_resources& resources)
{
  subtract_async(A, B, C, size, resources);
  resources.sync();
}

}  // namespace fast_deconv::matrix
