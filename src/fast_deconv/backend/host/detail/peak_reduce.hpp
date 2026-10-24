#pragma once

#include <cstdint>
#include <fast_deconv/matrix/peak.hpp>

// A lexicographic max on (value, -index), so the result is the same at any thread count.
// The initializer spells out matrix::kPeakIdentity: OpenMP lets it name only omp_priv and omp_orig.
#pragma omp declare reduction(peak_max : fast_deconv::matrix::peak : omp_out =      \
                                  fast_deconv::matrix::peak_max{}(omp_out, omp_in)) \
    initializer(omp_priv = fast_deconv::matrix::peak{.index = INT64_MAX, .value = -INFINITY})
