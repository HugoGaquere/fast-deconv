#pragma once
#include <cstdint>
#include <fast_deconv/core/host_memory_types.hpp>
#include <limits>
#include <stdexcept>
#include <string>

// Resolved by the include path: CMake puts backend/${FAST_DECONV_BACKEND}
// on it, and every backend provides this file.
#include <fd_backend/core/memory_types.hpp>

namespace fast_deconv::core {

/// Extents are int32 individually, but the plane product is not bounded by the type,
/// and cuBLAS/CUB/thrust call sites narrow a plane offset to int.
inline void check_plane_fits_int32(std::int64_t nrow, std::int64_t ncol, const char* what)
{
  if (nrow * ncol > std::numeric_limits<std::int32_t>::max())
    throw std::invalid_argument(std::string(what) + " plane " + std::to_string(nrow) + "x" + std::to_string(ncol) +
                                " exceeds the int32 index limit");
}

}  // namespace fast_deconv::core
