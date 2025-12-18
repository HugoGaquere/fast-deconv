#pragma once

#include <emu/cuda/device/mdspan.hpp>
#include <fast_deconv/core/concepts.hpp>
#include <fast_deconv/core/span_types.hpp>
#include <fast_deconv/core/stream_resources.hpp>
#include <fast_deconv/matrix/detail/subtract.cuh>

namespace fast_deconv::matrix {

namespace cpts = fast_deconv::core::cpts;

// inline void subtract_async(
//   const float* A, const float* B, float* C, size_t size, core::stream_resources& resources)
// {
//   detail::subtract_async(A, B, C, size, resources);
// }
//
// inline void subtract(
//   const float* A, const float* B, float* C, size_t size, core::stream_resources& resources)
// {
//   subtract_async(A, B, C, size, resources);
//   resources.sync();
// }

template <cpts::mdspan Mdspan>
inline void subtract_async(const Mdspan& A,
                           const Mdspan& B,
                           Mdspan& C,
                           core::stream_resources& resources)
{
  detail::subtract_async(A, B, C, resources);
}

template <cpts::mdspan Mdspan>
inline void subtract(const Mdspan& A, const Mdspan& B, Mdspan& C, core::stream_resources& resources)
{
  subtract_async(A, B, C, resources);
  resources.sync();
}

}  // namespace fast_deconv::matrix
