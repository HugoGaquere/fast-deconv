#pragma once
#include <cstdint>
#include <emu/detail/mdspan_types.hpp>
#include <limits>
#include <stdexcept>
#include <string>

namespace fast_deconv::core {

template <std::size_t N>
using dims = emu::dextents<std::int64_t, N>;

/// Spans index in int64, but the kernels and cuBLAS/CUB/thrust call sites still
/// hold a plane offset in an int, so every 2D grid has to fit one.
inline void check_plane_fits_int32(std::int64_t nrow, std::int64_t ncol, const char* what)
{
  if (nrow * ncol > std::numeric_limits<std::int32_t>::max())
    throw std::invalid_argument(std::string(what) + " plane " + std::to_string(nrow) + "x" + std::to_string(ncol) +
                                " exceeds the int32 index limit");
}

}  // namespace fast_deconv::core
