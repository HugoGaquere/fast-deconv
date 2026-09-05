#pragma once
#include <emu/detail/mdspan_types.hpp>
#include <emu/mdcontainer.hpp>
#include <emu/mdspan.hpp>
#include <fast_deconv/core/dims.hpp>

namespace fast_deconv::core {

// Host memory: caller-owned numpy arrays under any backend, and the CPU
// backend's working memory. Untagged, matching what emu's pybind casters produce.
template <typename T, std::size_t N>
using h_mdspan = emu::mdspan<T, dims<N>, emu::layout_right>;

template <typename T>
using host_span1d = h_mdspan<T, 1>;
template <typename T>
using host_span2d = h_mdspan<T, 2>;
template <typename T>
using host_span3d = h_mdspan<T, 3>;
template <typename T>
using host_span4d = h_mdspan<T, 4>;
template <typename T>
using host_span5d = h_mdspan<T, 5>;

// Owning host buffers; untagged like the spans, so one converts to host_spanNd.
template <typename T, std::size_t N>
using h_mdcontainer = emu::mdcontainer<T, dims<N>, emu::layout_right>;

template <typename T>
using host_cont1d = h_mdcontainer<T, 1>;
template <typename T>
using host_cont2d = h_mdcontainer<T, 2>;
template <typename T>
using host_cont3d = h_mdcontainer<T, 3>;
template <typename T>
using host_cont4d = h_mdcontainer<T, 4>;
template <typename T>
using host_cont5d = h_mdcontainer<T, 5>;

}  // namespace fast_deconv::core
