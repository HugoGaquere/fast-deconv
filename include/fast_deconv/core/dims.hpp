#pragma once
#include <cstdint>
#include <emu/detail/mdspan_types.hpp>

namespace fast_deconv::core {

template <std::size_t N>
using dims = emu::dextents<std::int32_t, N>;

}  // namespace fast_deconv::core
