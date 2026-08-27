#pragma once
#include <fast_deconv/core/memory_types.hpp>

namespace fast_deconv::core {

// Legacy spellings kept so existing call sites keep building while they migrate
// to span1d/cont1d. Delete once nothing names them.
template <typename T>
using host_vect = host_span1d<T>;
template <typename T>
using device_vect = span1d<T>;
template <typename T>
using device_cont = cont1d<T>;

}  // namespace fast_deconv::core
