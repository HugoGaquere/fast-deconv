#pragma once
#include <emu/detail/mdspan_types.hpp>
#include <emu/mdcontainer.hpp>
#include <emu/mdspan.hpp>
#include <fast_deconv/core/dims.hpp>

namespace fast_deconv::core {

// Host memory: caller-owned numpy arrays under any backend, and the CPU
// backend's working memory. Untagged, matching what emu's pybind casters produce.
template <typename T, std::size_t N>
using host_mdspan = emu::mdspan<T, dims<N>, emu::layout_right>;

template <typename T>
using host_span1d = host_mdspan<T, 1>;
template <typename T>
using host_span2d = host_mdspan<T, 2>;
template <typename T>
using host_span3d = host_mdspan<T, 3>;
template <typename T>
using host_span4d = host_mdspan<T, 4>;

// Owning host buffers; untagged like the spans, so one converts to host_spanNd.
template <typename T, std::size_t N>
using host_mdcontainer = emu::mdcontainer<T, dims<N>, emu::layout_right>;

template <typename T>
using host_cont1d = host_mdcontainer<T, 1>;
template <typename T>
using host_cont2d = host_mdcontainer<T, 2>;
template <typename T>
using host_cont3d = host_mdcontainer<T, 3>;
template <typename T>
using host_cont4d = host_mdcontainer<T, 4>;

}  // namespace fast_deconv::core
