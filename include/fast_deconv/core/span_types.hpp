#pragma once
#include <emu/cuda/device/mdspan.hpp>

namespace fast_deconv::core {

template <std::size_t N> using dims = emu::dextents<std::size_t, N>;

// Layout right
template <typename T, std::size_t N> using mdspan = emu::cuda::device::mdspan<T, dims<N>, emu::layout_right>;

// Generic typed span aliases (layout_right)
template <typename T> using span_1d = mdspan<T, 1>;
template <typename T> using span_2d = mdspan<T, 2>;
template <typename T> using span_3d = mdspan<T, 3>;
template <typename T> using span_4d = mdspan<T, 4>;
template <typename T> using span_5d = mdspan<T, 5>;
template <typename T> using span_6d = mdspan<T, 6>;

using device_vect_f   = mdspan<float, 1>;
using device_span2d_f = mdspan<float, 2>;
using device_span3d_f = mdspan<float, 3>;
using device_span4d_f = mdspan<float, 4>;
using device_span5d_f = mdspan<float, 5>;
using device_span6d_f = mdspan<float, 6>;

using device_vect_b   = mdspan<bool, 1>;
using device_span2d_b = mdspan<bool, 2>;
using device_span3d_b = mdspan<bool, 3>;
using device_span4d_b = mdspan<bool, 4>;
using device_span5d_b = mdspan<bool, 5>;
using device_span6d_b = mdspan<bool, 6>;

// Layout stride
template <typename T, std::size_t N> using mdspan_strided = emu::cuda::device::mdspan<T, dims<N>, emu::layout_stride>;

using device_vect_fs   = mdspan_strided<float, 1>;
using device_span2d_fs = mdspan_strided<float, 2>;
using device_span3d_fs = mdspan_strided<float, 3>;
using device_span4d_fs = mdspan_strided<float, 4>;
using device_span5d_fs = mdspan_strided<float, 5>;
using device_span6d_fs = mdspan_strided<float, 6>;

using device_vect_bs   = mdspan_strided<bool, 1>;
using device_span2d_bs = mdspan_strided<bool, 2>;
using device_span3d_bs = mdspan_strided<bool, 3>;
using device_span4d_bs = mdspan_strided<bool, 4>;
using device_span5d_bs = mdspan_strided<bool, 5>;
using device_span6d_bs = mdspan_strided<bool, 6>;


}  // namespace fast_deconv::core
