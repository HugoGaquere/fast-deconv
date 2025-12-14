#pragma once
#include <emu/cuda/device/mdspan.hpp>

namespace fast_deconv::core {

template <std::size_t N> using dims = emu::dextents<std::size_t, N>;
template <typename T, std::size_t N> using mdspan = emu::cuda::device::mdspan<T, dims<N>>;

template <typename T> using device_vect = mdspan<T, 1>;
template <typename T> using device_span2d = mdspan<T, 2>;
template <typename T> using device_span3d = mdspan<T, 3>;
template <typename T> using device_span4d = mdspan<T, 4>;
template <typename T> using device_span5d = mdspan<T, 5>;
template <typename T> using device_span6d = mdspan<T, 6>;

using device_vect_f   = device_vect<float>;
using device_span2d_f = device_span2d<float>;
using device_span3d_f = device_span3d<float>;
using device_span4d_f = device_span4d<float>;
using device_span5d_f = device_span5d<float>;
using device_span6d_f = device_span6d<float>;

using device_vect_b   = device_vect<bool>;
using device_span2d_b = device_span2d<bool>;
using device_span3d_b = device_span3d<bool>;
using device_span4d_b = device_span4d<bool>;
using device_span5d_b = device_span5d<bool>;
using device_span6d_b = device_span6d<bool>;


}  // namespace fast_deconv::core
