#pragma once

#include <fast_deconv/core/concepts.hpp>
#include <fast_deconv/core/span_types.hpp>

namespace fast_deconv::util {

template <core::cpts::mdspan Mdspan>
__device__ __forceinline__ auto linear_to_indices(typename Mdspan::size_type lin, const Mdspan& m)
{
  using size_t_ = typename Mdspan::size_type;
  using index_t = typename Mdspan::index_type;
  constexpr int R = Mdspan::rank();

  std::array<index_t, R> idx{};

#pragma unroll
  for (int d = R - 1; d >= 0; --d) {
    const auto e = static_cast<size_t_>(m.extent(d));
    const size_t_ r = lin % e;
    lin /= e;
    idx[d] = static_cast<index_t>(r);
  }
  return idx;
}

inline auto unravel_index_2D(int flat_index, int width) -> std::pair<int, int>
{
  const int y = flat_index / width;
  const int x = flat_index % width;
  return {y, x};
}

}  // namespace fast_deconv::util
