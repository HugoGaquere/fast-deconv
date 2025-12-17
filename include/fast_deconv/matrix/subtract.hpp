#pragma once

#include <emu/cuda/device/mdspan.hpp>
#include <fast_deconv/core/span_types.hpp>
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

template <typename Mdspan>
inline void subtract_async(const Mdspan& A,
                           const Mdspan& B,
                           Mdspan& C,
                           core::stream_resources& resources)
{
  using layout_type = typename Mdspan::layout_type;
  if constexpr (std::is_same_v<layout_type, emu::layout_stride>)
    throw std::runtime_error("Strided mdspan not supported");

  // if constexpr (std::is_same_v<layout_type, std::layout_stride>)
  //   throw std::runtime_error("Strided mdspan not supported");

  detail::subtract_async(A, B, C, resources);
}

template <typename Mdspan>
inline void subtract(const Mdspan& A, const Mdspan& B, Mdspan& C, core::stream_resources& resources)
{
  subtract_async(A, B, C, resources);
  resources.sync();
}

}  // namespace fast_deconv::matrix
