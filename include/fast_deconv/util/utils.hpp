#pragma once

#include <cstdint>
#include <fast_deconv/common/region.hpp>

namespace fast_deconv::util {

inline auto unravel_index_2D(std::int64_t flat_index, std::int64_t ncol) -> common::index2d
{
  return {flat_index / ncol, flat_index % ncol};
}

}  // namespace fast_deconv::util
