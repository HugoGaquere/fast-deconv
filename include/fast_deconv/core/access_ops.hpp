#pragma once

#include <fast_deconv/core/access_policy.hpp>

#include <cstddef>
#include <type_traits>

namespace fast_deconv::core {


template <AccessType LoadPolicy, typename T>
__device__ T load(const T* ptr) noexcept
{

}

template <AccessType StorePolicy, typename T>
__device__ void store(T* ptr, T value) noexcept
{

}

}  // namespace fast_deconv::core
