#pragma once

#include <cstdint>

namespace fast_deconv::matrix {

/// Value and flat row-major index of an image maximum.
struct peak {
  float value;
  std::int64_t index;
};

}  // namespace fast_deconv::matrix
