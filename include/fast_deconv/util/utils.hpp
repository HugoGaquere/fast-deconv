#pragma once

#include <cstdint>
#include <fast_deconv/core/region.hpp>

namespace fast_deconv::util {

/// (row, col) of @p flat_index in a row-major image with @p ncol columns.
inline auto unravel_index_2D(std::int64_t flat_index, std::int64_t ncol) -> core::index2d
{
  return {flat_index / ncol, flat_index % ncol};
}

}  // namespace fast_deconv::util
