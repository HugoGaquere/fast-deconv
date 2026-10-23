#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cub/device/device_reduce.cuh>
#include <fast_deconv/matrix/stats.hpp>
#include <fast_deconv/util/cuda_macros.hpp>

namespace fast_deconv::matrix {

namespace {

// Every pixel feeds the standard deviation, masked or not: DDFacet's np.std(MeanDirty) stop threshold.
struct masked_stats_op {
  const float* data;
  const bool* mask;
  bool use_abs;

  __host__ __device__ __forceinline__ stats_acc operator()(std::int64_t idx) const
  {
    const float v = data[idx];
    const float max_v = mask[idx] ? -INFINITY : (use_abs ? fabsf(v) : v);
    return {max_v, v, v * v, 1};
  }
};

struct stats_combine {
  __host__ __device__ __forceinline__ stats_acc operator()(const stats_acc& a, const stats_acc& b) const
  {
    return {fmaxf(a.max_v, b.max_v), a.sum + b.sum, a.sum_sq + b.sum_sq, a.count + b.count};
  }
};

using counting_it = thrust::counting_iterator<std::int64_t>;
using transform_it = thrust::transform_iterator<masked_stats_op, counting_it>;

inline transform_it make_iter(const float* data, const bool* mask, bool use_abs)
{
  return thrust::make_transform_iterator(counting_it{0}, masked_stats_op{data, mask, use_abs});
}

// -inf is the masked-pixel fill, so the max sentinel ties with it instead of
// beating it: an all-masked image reports -inf, not a finite floor.
constexpr stats_acc kIdentity = {-INFINITY, 0.f, 0.f, 0};

}  // namespace

stats_impl::stats_impl(const core::exec_ctx& ctx, std::int64_t n_elements, bool use_abs)
    : ctx_(ctx), n_elements_(n_elements), use_abs_(use_abs)
{
  d_acc_ = ctx_.alloc_ptr<stats_acc>(1);

  // Query temp-storage bytes against the same iterator/op/state types we will
  // use at call time. The pointers held by the iterator are unused during the
  // sizing query.
  auto it_query = make_iter(nullptr, nullptr, use_abs_);
  CHECK_CUDA(cub::DeviceReduce::Reduce(nullptr, temp_bytes_, it_query, d_acc_.get(), n_elements_, stats_combine{},
                                       kIdentity, ctx_.cuda_stream));

  d_temp_ = ctx_.alloc_ptr<std::byte>(core::narrow<std::int64_t>(std::max<std::size_t>(temp_bytes_, 1)));
}

stats_acc stats_impl::accumulate(core::span2d<const float> data, core::span2d<const bool> mask)
{
  auto it = make_iter(data.data_handle(), mask.data_handle(), use_abs_);
  CHECK_CUDA(cub::DeviceReduce::Reduce(d_temp_.get(), temp_bytes_, it, d_acc_.get(), n_elements_, stats_combine{},
                                       kIdentity, ctx_.cuda_stream));

  stats_acc h_acc{};
  CHECK_CUDA(cudaMemcpyAsync(&h_acc, d_acc_.get(), sizeof(stats_acc), cudaMemcpyDeviceToHost, ctx_.cuda_stream));
  ctx_.wait();
  return h_acc;
}

}  // namespace fast_deconv::matrix
