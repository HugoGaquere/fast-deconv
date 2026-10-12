#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cub/device/device_reduce.cuh>
#include <fast_deconv/matrix/argmax.hpp>
#include <fast_deconv/util/cuda_macros.hpp>

namespace fast_deconv::matrix {

namespace {

struct criterion_peak {
  const float* data;
  peak_criterion criterion;

  __host__ __device__ __forceinline__ peak operator()(std::int64_t i) const
  {
    const float x = data[i];
    return {.index = i, .value = criterion(x, i), .signed_value = x};
  }
};

// -inf ties with a masked pixel, and the largest index loses that tie.
constexpr peak kReduceIdentity{.index = INT64_MAX, .value = -INFINITY};

}  // namespace

peak find_peak(const core::exec_ctx& ctx, core::span2d<const float> data, peak_criterion criterion)
{
  assert(data.is_exhaustive());

  // A reduce rather than ArgMax, so the signed pixel travels with the winning criterion.
  const auto it = thrust::make_transform_iterator(thrust::counting_iterator<std::int64_t>{0},
                                                  criterion_peak{data.data_handle(), criterion});
  const int n = static_cast<int>(data.size());
  auto d_result = ctx.alloc_ptr_async<peak>(1);

  std::size_t temp_bytes = 0;
  CHECK_CUDA(cub::DeviceReduce::Reduce(nullptr, temp_bytes, it, d_result.get(), n, peak_max{}, kReduceIdentity,
                                       ctx.cuda_stream));
  auto d_temp = ctx.alloc_ptr_async<std::byte>(std::max<std::size_t>(temp_bytes, 1));
  CHECK_CUDA(cub::DeviceReduce::Reduce(d_temp.get(), temp_bytes, it, d_result.get(), n, peak_max{}, kReduceIdentity,
                                       ctx.cuda_stream));

  peak result{};
  CHECK_CUDA(cudaMemcpyAsync(&result, d_result.get(), sizeof(peak), cudaMemcpyDeviceToHost, ctx.cuda_stream));
  ctx.wait();
  return result;
}

}  // namespace fast_deconv::matrix
