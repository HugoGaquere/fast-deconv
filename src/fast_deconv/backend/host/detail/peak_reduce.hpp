#pragma once

#include <cstdint>
#include <fast_deconv/matrix/peak.hpp>
#include <limits>

namespace fast_deconv::detail {

/// Max by value; ties go to the smaller index, matching std::max_element and CUB.
inline matrix::peak max_by_value(const matrix::peak& a, const matrix::peak& b)
{
  if (a.value > b.value) return a;
  if (b.value > a.value) return b;
  return (a.index <= b.index) ? a : b;
}

// True identity: -inf ties with a masked pixel, and the largest index loses that tie.
inline constexpr matrix::peak kPeakIdentity{-std::numeric_limits<float>::infinity(), INT64_MAX};

}  // namespace fast_deconv::detail

// A lexicographic max on (value, -index), so the result is the same at any thread count.
#pragma omp declare reduction(peak_max : fast_deconv::matrix::peak : omp_out = fast_deconv::detail::max_by_value( \
                                      omp_out, omp_in)) initializer(omp_priv = fast_deconv::detail::kPeakIdentity)
