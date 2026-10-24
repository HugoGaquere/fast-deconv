#pragma once
#include <concepts>
#include <cstdint>
#include <emu/detail/mdspan_types.hpp>
#include <stdexcept>
#include <string>
#include <utility>

namespace fast_deconv::core {

template <std::size_t N>
using dims = emu::dextents<std::int64_t, N>;

/**
 * @brief The one narrowing conversion.
 *
 * Counts are int64, and a third-party API (cuBLAS, cuFFT, a launch config, a size_t shape) that takes less
 * gets its value through here, never through a bare cast.
 *
 * @param[in] v  Value to convert.
 * @return @p v as @p To.
 * @throws std::overflow_error If @p v does not fit in @p To.
 */
template <std::integral To, std::integral From>
To narrow(From v)
{
  if (!std::in_range<To>(v))
    throw std::overflow_error("narrow: " + std::to_string(v) + " does not fit the target type");
  return static_cast<To>(v);
}

constexpr std::int64_t ceil_div(std::int64_t a, std::int64_t b) { return (a + b - 1) / b; }

}  // namespace fast_deconv::core
