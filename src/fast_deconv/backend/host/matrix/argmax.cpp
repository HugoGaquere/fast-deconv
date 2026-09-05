#include <cassert>
#include <cstddef>
#include <cstdint>
#include <fast_deconv/matrix/argmax.hpp>
#include <limits>

#include "../detail/peak_reduce.hpp"

namespace fast_deconv::matrix {

void argmax_ctx::run_async(core::span2d<float> data)
{
  assert(data.is_exhaustive());
  assert(data.size() == n_elements_);

  if (n_elements_ == 0) {
    last_ = {-std::numeric_limits<float>::infinity(), 0};
    return;
  }

  // Keeps the first maximum, matching CUB's smaller-index tie-break, at any thread count.
  const float* d = data.data_handle();
  peak best = detail::kPeakIdentity;
#pragma omp parallel for reduction(peak_max : best)
  for (std::size_t i = 0; i < n_elements_; i++) best = detail::max_by_value(best, {d[i], static_cast<std::int64_t>(i)});

  last_ = best;
}

peak argmax_ctx::run(core::span2d<float> data)
{
  run_async(data);
  return last_;
}

}  // namespace fast_deconv::matrix
