#pragma once

#include <cstdint>

namespace fast_deconv::core {

/// A 2D index into a row-major array. Ordered row-then-column, matching span2d's extent order.
struct index2d {
  std::int64_t row, col;
};

/// (row, col) of @p flat_index in a row-major image with @p ncol columns.
inline index2d unravel_index_2D(std::int64_t flat_index, std::int64_t ncol)
{
  return {flat_index / ncol, flat_index % ncol};
}

}  // namespace fast_deconv::core
