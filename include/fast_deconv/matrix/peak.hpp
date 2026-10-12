#pragma once

#include <cmath>
#include <cstdint>
#include <fast_deconv/core/macro.hpp>

namespace fast_deconv::matrix {

/// Value and flat row-major index of an image maximum.
struct peak {
  std::int64_t index;
  float value;                // what the search ranked by: the peak_criterion result, |x| when absolute
  float signed_value = 0.0f;  // the pixel itself, sign kept;
};

/// Max by value; ties go to the smaller index, matching std::max_element and CUB.
struct peak_max {
  FD_HOST_DEVICE peak operator()(const peak& a, const peak& b) const
  {
    if (a.value > b.value) return a;
    if (b.value > a.value) return b;
    return (a.index <= b.index) ? a : b;
  }
};

/// Value a peak search ranks pixel i by: -inf where masked, |x| when absolute.
struct peak_criterion {
  const bool* mask;  // nullptr = no mask
  bool absolute;

  FD_HOST_DEVICE float operator()(float x, std::int64_t i) const
  {
    if (mask && mask[i]) return -INFINITY;
    return absolute && x < 0.0f ? -x : x;
  }
};

}  // namespace fast_deconv::matrix
