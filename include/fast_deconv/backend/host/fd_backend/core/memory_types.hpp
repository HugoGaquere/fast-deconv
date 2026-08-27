#pragma once

#include <fast_deconv/core/host_memory_types.hpp>

namespace fast_deconv::core {

// Backend-resident memory for this build: host memory, so the generic aliases
// are the host ones. No separate device_* family exists in a host build.
template <typename T, std::size_t N>
using mdspan = h_mdspan<T, N>;
template <typename T, std::size_t N>
using mdcontainer = h_mdcontainer<T, N>;

template <typename T>
using span1d = host_span1d<T>;
template <typename T>
using span2d = host_span2d<T>;
template <typename T>
using span3d = host_span3d<T>;
template <typename T>
using span4d = host_span4d<T>;
template <typename T>
using span5d = host_span5d<T>;

template <typename T>
using cont1d = host_cont1d<T>;
template <typename T>
using cont2d = host_cont2d<T>;
template <typename T>
using cont3d = host_cont3d<T>;
template <typename T>
using cont4d = host_cont4d<T>;

}  // namespace fast_deconv::core
