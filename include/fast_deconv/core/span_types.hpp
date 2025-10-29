#pragma once
#include <cuda/std/mdspan>

namespace fast_deconv::core {

using Extents6D = cuda::std::dextents<int, 6>;
using Extents5D = cuda::std::dextents<int, 5>;
using Extents4D = cuda::std::dextents<int, 4>;
using Extents3D = cuda::std::dextents<int, 3>;
using Extents2D = cuda::std::dextents<int, 2>;
using Extents1D = cuda::std::dextents<int, 1>;

template<typename T> using Span6d   = cuda::std::mdspan<T, Extents6D>;
template<typename T> using Span5d   = cuda::std::mdspan<T, Extents5D>;
template<typename T> using Span4d   = cuda::std::mdspan<T, Extents4D>;
template<typename T> using Span3d   = cuda::std::mdspan<T, Extents3D>;
template<typename T> using Span2d   = cuda::std::mdspan<T, Extents2D>;
template<typename T> using SpanVect = cuda::std::mdspan<T, Extents1D>;

template <typename T, typename Extent>
inline auto make_span(T* ptr, Extent ext)
{
  return cuda::std::mdspan<T, Extent>(ptr, ext);
}

}  // namespace fast_deconv::core
