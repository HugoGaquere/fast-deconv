#pragma once

#include <fast_deconv/core/concepts.hpp>
#include <fast_deconv/core/span_types.hpp>

#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace fast_deconv::util {

template <core::cpts::mdspan Mdspan>
constexpr bool inner_unit_stride(const Mdspan& m)
{
  if constexpr (core::cpts::is_layout_right<Mdspan>) {
    return true;
  } else if constexpr (core::cpts::is_layout_left<Mdspan>) {
    return false;
  } else {
    constexpr int R = Mdspan::rank();
    return m.mapping().stride(R - 1) == 1;
  }
}

template <core::cpts::mdspan... Mdspans>
constexpr bool all_inner_unit_stride(const Mdspans&... mdspans)
{
  return (inner_unit_stride(mdspans) && ...);
}

template <class Ptr>
requires std::is_pointer_v<Ptr> constexpr bool is_aligned_for_vec(Ptr p,
                                                                  std::size_t vec_bytes) noexcept
{
  // vec_bytes should be power-of-two for this fast check.
  return (reinterpret_cast<std::uintptr_t>(p) & (vec_bytes - 1)) == 0;
}

template <core::cpts::mdspan Mdspan>
constexpr bool is_aligned_for_vec(const Mdspan& m, std::size_t vec_bytes) noexcept
{
  auto p = m.data_handle();
  return is_aligned_for_vec(p, vec_bytes);
}

template <core::cpts::mdspan... Mdspans>
constexpr bool all_aligned_for_vec(std::size_t vec_bytes, const Mdspans&... mdspans)
{
  return (is_aligned_for_vec(mdspans, vec_bytes) && ...);
}

}  // namespace fast_deconv::util
