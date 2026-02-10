#pragma once

#include <emu/cuda/device/mdspan.hpp>
#include <fast_deconv/core/access_policy.hpp>
#include <fast_deconv/core/concepts.hpp>
#include <fast_deconv/core/dispatcher.hpp>
#include <fast_deconv/core/kernel_traits.hpp>
#include <fast_deconv/core/span_types.hpp>
#include <fast_deconv/core/stream_resources.hpp>
#include <fast_deconv/matrix/detail/subtract.cuh>

#include <cstdint>

namespace fast_deconv::matrix {

template <core::cpts::mdspan Mdspan>
inline void subtract_async(const Mdspan& A,
                           const Mdspan& B,
                           Mdspan& C,
                           core::stream_resources& resources)
{
  // detail::subtract_async(access_policy, resources, A, B, C);
  core::dispatch<core::subtract_kernel_tag>(resources, detail::subtract_async<Mdspan>, A, B, C);
}

template <core::cpts::mdspan Mdspan>
inline void subtract(const Mdspan& A, const Mdspan& B, Mdspan& C, core::stream_resources& resources)
{
  subtract_async(A, B, C, resources);
  // resources.sync();
}

}  // namespace fast_deconv::matrix
