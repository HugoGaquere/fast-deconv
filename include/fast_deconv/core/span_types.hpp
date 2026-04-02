#pragma once
#include <emu/mdspan.hpp>
#include <emu/cuda/device/mdspan.hpp>

namespace fast_deconv::core {

template <std::size_t N>
using dims = emu::dextents<std::size_t, N>;

// ================================================================== //
//                     Device MDSpan
// ================================================================== //

// Layout right (row-major, C-order)
template <typename T, std::size_t N>
using mdspan = emu::cuda::device::mdspan<T, dims<N>, emu::layout_right>;

template <typename T>
using device_vect = mdspan<T, 1>;
template <typename T>
using device_span2d = mdspan<T, 2>;
template <typename T>
using device_span3d = mdspan<T, 3>;
template <typename T>
using device_span4d = mdspan<T, 4>;
template <typename T>
using device_span5d = mdspan<T, 5>;
template <typename T>
using device_span6d = mdspan<T, 6>;

// Layout left (column-major, Fortran-order)
template <typename T, std::size_t N>
using mdspan_F = emu::cuda::device::mdspan<T, dims<N>, emu::layout_left>;

template <typename T>
using device_vect_F = mdspan_F<T, 1>;
template <typename T>
using device_span2d_F = mdspan_F<T, 2>;
template <typename T>
using device_span3d_F = mdspan_F<T, 3>;
template <typename T>
using device_span4d_F = mdspan_F<T, 4>;
template <typename T>
using device_span5d_F = mdspan_F<T, 5>;
template <typename T>
using device_span6d_F = mdspan_F<T, 6>;

// Layout stride
template <typename T, std::size_t N>
using mdspan_S = emu::cuda::device::mdspan<T, dims<N>, emu::layout_stride>;

template <typename T>
using device_vect_S = mdspan_S<T, 1>;
template <typename T>
using device_span2d_S = mdspan_S<T, 2>;
template <typename T>
using device_span3d_S = mdspan_S<T, 3>;
template <typename T>
using device_span4d_S = mdspan_S<T, 4>;
template <typename T>
using device_span5d_S = mdspan_S<T, 5>;
template <typename T>
using device_span6d_S = mdspan_S<T, 6>;

// ================================================================== //
//                     Host MDSpan
// ================================================================== //

// Layout right (row-major, C-order)
template <typename T, std::size_t N>
using h_mdspan = emu::mdspan<T, dims<N>, emu::layout_right>;

template <typename T>
using host_vect = h_mdspan<T, 1>;
template <typename T>
using host_span2d = h_mdspan<T, 2>;
template <typename T>
using host_span3d = h_mdspan<T, 3>;
template <typename T>
using host_span4d = h_mdspan<T, 4>;
template <typename T>
using host_span5d = h_mdspan<T, 5>;
template <typename T>
using host_span6d = h_mdspan<T, 6>;

// Layout left (column-major, Fortran-order)
template <typename T, std::size_t N>
using h_mdspan_F = emu::mdspan<T, dims<N>, emu::layout_left>;

template <typename T>
using host_vect_F = h_mdspan_F<T, 1>;
template <typename T>
using host_span2d_F = h_mdspan_F<T, 2>;
template <typename T>
using host_span3d_F = h_mdspan_F<T, 3>;
template <typename T>
using host_span4d_F = h_mdspan_F<T, 4>;
template <typename T>
using host_span5d_F = h_mdspan_F<T, 5>;
template <typename T>
using host_span6d_F = h_mdspan_F<T, 6>;

// Layout stride
template <typename T, std::size_t N>
using h_mdspan_S = emu::mdspan<T, dims<N>, emu::layout_stride>;

template <typename T>
using host_vect_S = h_mdspan_S<T, 1>;
template <typename T>
using host_span2d_S = h_mdspan_S<T, 2>;
template <typename T>
using host_span3d_S = h_mdspan_S<T, 3>;
template <typename T>
using host_span4d_S = h_mdspan_S<T, 4>;
template <typename T>
using host_span5d_S = h_mdspan_S<T, 5>;
template <typename T>
using host_span6d_S = h_mdspan_S<T, 6>;


}  // namespace fast_deconv::core
