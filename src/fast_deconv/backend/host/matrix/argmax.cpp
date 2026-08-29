#include <algorithm>
#include <cassert>
#include <cfloat>
#include <cstdint>
#include <fast_deconv/matrix/argmax.hpp>

namespace fast_deconv::matrix {

void argmax_ctx::run_async(core::span2d<float> data)
{
  assert(data.is_exhaustive());
  assert(data.size() == n_elements_);

  if (n_elements_ == 0) {
    last_ = {-FLT_MAX, 0};
    return;
  }

  // std::max_element keeps the first maximum, matching CUB's smaller-index tie-break.
  const float* begin = data.data_handle();
  const float* best = std::max_element(begin, begin + n_elements_);
  last_ = {*best, static_cast<std::int64_t>(best - begin)};
}

peak argmax_ctx::run(core::span2d<float> data)
{
  run_async(data);
  return last_;
}

}  // namespace fast_deconv::matrix
