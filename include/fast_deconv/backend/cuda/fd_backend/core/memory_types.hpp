#pragma once

#include <emu/cuda/device/mdcontainer.hpp>
#include <emu/cuda/device/mdspan.hpp>
#include <emu/cuda/device/span.hpp>
#include <fast_deconv/core/dims.hpp>

namespace fast_deconv::core {

// Non-owning device views
template <typename T, std::size_t N>
using device_mdspan = emu::cuda::device::mdspan<T, dims<N>, emu::layout_right>;

template <typename T>
using device_span1d = device_mdspan<T, 1>;
template <typename T>
using device_span2d = device_mdspan<T, 2>;
template <typename T>
using device_span3d = device_mdspan<T, 3>;
template <typename T>
using device_span4d = device_mdspan<T, 4>;
template <typename T>
using device_span5d = device_mdspan<T, 5>;

// Owning device containers (RAII, refcounted via emu::capsule)
template <typename T, std::size_t N>
using device_mdcontainer = emu::cuda::device::mdcontainer<T, dims<N>, emu::layout_right>;

template <typename T>
using device_cont1d = device_mdcontainer<T, 1>;
template <typename T>
using device_cont2d = device_mdcontainer<T, 2>;
template <typename T>
using device_cont3d = device_mdcontainer<T, 3>;
template <typename T>
using device_cont4d = device_mdcontainer<T, 4>;
template <typename T>
using device_cont5d = device_mdcontainer<T, 5>;

// Backend-resident memory for this build: device memory.
template <typename T, std::size_t N>
using mdspan = device_mdspan<T, N>;
template <typename T, std::size_t N>
using mdcontainer = device_mdcontainer<T, N>;

template <typename T>
using span1d = device_span1d<T>;
template <typename T>
using span2d = device_span2d<T>;
template <typename T>
using span3d = device_span3d<T>;
template <typename T>
using span4d = device_span4d<T>;
template <typename T>
using span5d = device_span5d<T>;

template <typename T>
using cont1d = device_cont1d<T>;
template <typename T>
using cont2d = device_cont2d<T>;
template <typename T>
using cont3d = device_cont3d<T>;
template <typename T>
using cont4d = device_cont4d<T>;
template <typename T>
using cont5d = device_cont5d<T>;

}  // namespace fast_deconv::core
