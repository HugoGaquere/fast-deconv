#pragma once

#include <cstdint>
#include <fast_deconv/common/region.hpp>

namespace fast_deconv::util {

inline auto unravel_index_2D(std::int64_t flat_index, int ncol) -> common::index2d
{
  const auto row = flat_index / ncol;
  const auto col = flat_index % ncol;
  return {static_cast<int>(row), static_cast<int>(col)};
}

}  // namespace fast_deconv::util
