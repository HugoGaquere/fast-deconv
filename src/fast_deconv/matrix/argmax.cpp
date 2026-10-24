#include <cassert>
#include <cstdint>
#include <fast_deconv/core/profiler.hpp>
#include <fast_deconv/matrix/argmax.hpp>
#include <fd_backend/core/reduce.hpp>

namespace fast_deconv::matrix {

namespace {

// Pixel i as a candidate: ranked by the criterion, the signed pixel carried along.
struct criterion_peak {
  const float* data;
  peak_criterion criterion;

  FD_HOST_DEVICE peak operator()(std::int64_t i) const
  {
    const float x = data[i];
    return {.index = i, .value = criterion(x, i), .signed_value = x};
  }
};

}  // namespace

peak find_peak(const core::exec_ctx& ctx, core::span2d<const float> data, peak_criterion criterion)
{
  FD_PROFILE_FN();
  assert(data.is_exhaustive());
  return core::reduce(ctx, data.mapping().required_span_size(), criterion_peak{data.data_handle(), criterion},
                      peak_max{}, kPeakIdentity);
}

}  // namespace fast_deconv::matrix
