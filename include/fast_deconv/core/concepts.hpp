#pragma once
#include <emu/concepts.hpp>
#include <emu/cuda/device/mdspan.hpp>

#include <type_traits>

namespace fast_deconv::core::cpts {

template <typename T>
using layout_type = T::layout_type;

template <typename T>
concept is_layout_right = std::is_same_v<layout_type<T>, emu::layout_right>;
template <typename T>
concept is_layout_left = std::is_same_v<layout_type<T>, emu::layout_left>;
template <typename T>
concept is_layout_stride = std::is_same_v<layout_type<T>, emu::layout_stride>;

template <typename T>
concept mdspan = emu::cuda::device::cpts::mdspan<T>;

}  // namespace fast_deconv::core::cpts
