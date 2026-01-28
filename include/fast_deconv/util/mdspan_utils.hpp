#pragma once

#include <fast_deconv/core/concepts.hpp>
#include <fast_deconv/core/span_types.hpp>

#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace fast_deconv::util {

template <core::cpts::mdspan Mdspan>
__device__ __forceinline__ auto linear_to_indices(typename Mdspan::size_type lin, const Mdspan& m)
{
  using size_t_   = typename Mdspan::size_type;
  using index_t   = typename Mdspan::index_type;
  constexpr int R = Mdspan::rank();

  std::array<index_t, R> idx{};

#pragma unroll
  for (int d = R - 1; d >= 0; --d) {
    const auto e    = static_cast<size_t_>(m.extent(d));
    const size_t_ r = lin % e;
    lin /= e;
    idx[d] = static_cast<index_t>(r);
  }
  return idx;
}

// template <core::cpts::mdspan Mdspan>
// constexpr bool inner_unit_stride(const Mdspan& m)
// {
//   if constexpr (core::cpts::is_layout_right<Mdspan>) {
//     return true;
//   } else if constexpr (core::cpts::is_layout_left<Mdspan>) {
//     return false;
//   } else {
//     constexpr int R = Mdspan::rank();
//     return m.mapping().stride(R - 1) == 1;
//   }
// }
//
// template <core::cpts::mdspan... Mdspans>
// constexpr bool all_inner_unit_stride(const Mdspans&... mdspans)
// {
//   return (inner_unit_stride(mdspans) && ...);
// }
//
// template <class Ptr>
// requires std::is_pointer_v<Ptr> constexpr bool is_aligned_for_vec(Ptr p,
//                                                                   std::size_t vec_bytes) noexcept
// {
//   // vec_bytes should be power-of-two for this fast check.
//   return (reinterpret_cast<std::uintptr_t>(p) & (vec_bytes - 1)) == 0;
// }
//
// template <core::cpts::mdspan Mdspan>
// constexpr bool is_aligned_for_vec(const Mdspan& m, std::size_t vec_bytes) noexcept
// {
//   auto p = m.data_handle();
//   return is_aligned_for_vec(p, vec_bytes);
// }
//
// template <core::cpts::mdspan... Mdspans>
// constexpr bool all_aligned_for_vec(std::size_t vec_bytes, const Mdspans&... mdspans)
// {
//   return (is_aligned_for_vec(mdspans, vec_bytes) && ...);
// }

}  // namespace fast_deconv::util
