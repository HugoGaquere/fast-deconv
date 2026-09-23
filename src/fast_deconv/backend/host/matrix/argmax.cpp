#include <cassert>
#include <cstddef>
#include <cstdint>
#include <fast_deconv/core/profiler.hpp>
#include <fast_deconv/matrix/argmax.hpp>
#include <limits>

#include "../detail/peak_reduce.hpp"

namespace fast_deconv::matrix {

peak find_peak(const core::exec_ctx& ctx, core::span2d<const float> data, peak_criterion criterion)
{
  FD_PROFILE_FN();
  assert(data.is_exhaustive());

  const float* d = data.data_handle();
  const auto n = static_cast<std::int64_t>(data.size());
  peak best = detail::kPeakIdentity;
#pragma omp parallel for reduction(peak_max : best)
  for (std::int64_t i = 0; i < n; i++)
    best = detail::max_by_value(best, {.index = i, .value = criterion(d[i], i), .signed_value = d[i]});

  return best;
}

}  // namespace fast_deconv::matrix
