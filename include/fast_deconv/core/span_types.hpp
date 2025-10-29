#pragma once
#include <cuda/std/mdspan>

namespace fast_deconv::core {

using span_idx_t = int;

using extents_6d = cuda::std::dextents<span_idx_t, 6>;
using extents_5d = cuda::std::dextents<span_idx_t, 5>;
using extents_4d = cuda::std::dextents<span_idx_t, 4>;
using extents_3d = cuda::std::dextents<span_idx_t, 3>;
using extents_2d = cuda::std::dextents<span_idx_t, 2>;
using extents_1d = cuda::std::dextents<span_idx_t, 1>;

template<typename T> using span_6d   = cuda::std::mdspan<T, extents_6d>;
template<typename T> using span_5d   = cuda::std::mdspan<T, extents_5d>;
template<typename T> using span_4d   = cuda::std::mdspan<T, extents_4d>;
template<typename T> using span_3d   = cuda::std::mdspan<T, extents_3d>;
template<typename T> using span_2d   = cuda::std::mdspan<T, extents_2d>;
template<typename T> using span_vect = cuda::std::mdspan<T, extents_1d>;

template <typename T, typename Extent>
inline auto make_mdspan(T* ptr, Extent ext)
{
  return cuda::std::mdspan<T, Extent>(ptr, ext);
}

template <typename T, typename Extent>
inline auto make_mdspan_from_cupy(T* ptr, Extent ext)
{
  return cuda::std::mdspan<T, Extent>(ptr, ext);
}


}  // namespace fast_deconv::core
