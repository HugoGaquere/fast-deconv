#pragma once

#include <cstdint>
#include <utility>

namespace fast_deconv::util {

inline auto unravel_index_2D(std::int64_t flat_index, int width) -> std::pair<int, int>
{
  const auto y = flat_index / width;
  const auto x = flat_index % width;
  return {static_cast<int>(y), static_cast<int>(x)};
}

}  // namespace fast_deconv::util
