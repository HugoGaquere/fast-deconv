#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>

#include <algorithm>
#include <cassert>
#include <cfloat>
#include <cmath>
#include <cub/device/device_reduce.cuh>
#include <fast_deconv/matrix/stats.hpp>
#include <fast_deconv/util/cuda_macros.hpp>

namespace fast_deconv::matrix {

namespace {

/// Per-pixel transform: the mask excludes pixels from the max only. Every
/// pixel contributes to the RMS so the noise estimate is computed over the
/// whole image, matching DDFacet's `np.std(MeanDirty)` stop-threshold RMS.
struct masked_stats_op {
  const float* data;
  const bool* mask;
  bool use_abs;

  __host__ __device__ __forceinline__ stats_acc operator()(int idx) const
  {
    const float v = data[idx];
    const float max_v = mask[idx] ? -FLT_MAX : (use_abs ? fabsf(v) : v);
    return {max_v, v, v * v, 1};
  }
};

struct stats_combine {
  __host__ __device__ __forceinline__ stats_acc operator()(const stats_acc& a, const stats_acc& b) const
  {
    return {fmaxf(a.max_v, b.max_v), a.sum + b.sum, a.sum_sq + b.sum_sq, a.count + b.count};
  }
};

using counting_it = thrust::counting_iterator<int>;
using transform_it = thrust::transform_iterator<masked_stats_op, counting_it>;

inline transform_it make_iter(const float* data, const bool* mask, bool use_abs)
{
  return thrust::make_transform_iterator(counting_it{0}, masked_stats_op{data, mask, use_abs});
}

constexpr stats_acc kIdentity = {-FLT_MAX, 0.f, 0.f, 0};

}  // namespace

stats_ctx::stats_ctx(const core::exec_ctx& ctx, std::size_t n_elements, bool use_abs)
    : ctx_(ctx), n_elements_(n_elements), use_abs_(use_abs)
{
  d_state_ = ctx_.alloc_ptr_async<stats_acc>(1);

  // Query temp-storage bytes against the same iterator/op/state types we will
  // use at call time. The pointers held by the iterator are unused during the
  // sizing query.
  auto it_query = make_iter(nullptr, nullptr, use_abs_);
  CHECK_CUDA(cub::DeviceReduce::Reduce(nullptr, temp_bytes_, it_query, d_state_.get(), static_cast<int>(n_elements_),
                                       stats_combine{}, kIdentity, ctx_.cuda_stream));

  d_temp_ = ctx_.alloc_ptr_async<std::byte>(temp_bytes_);
}

void stats_ctx::run_async(core::span2d<float> data, core::span2d<bool> mask)
{
  assert(data.is_exhaustive() && mask.is_exhaustive());
  assert(data.extent(0) == mask.extent(0) && data.extent(1) == mask.extent(1));
  assert(data.size() == n_elements_);

  auto it = make_iter(data.data_handle(), mask.data_handle(), use_abs_);
  CHECK_CUDA(cub::DeviceReduce::Reduce(d_temp_.get(), temp_bytes_, it, d_state_.get(), static_cast<int>(n_elements_),
                                       stats_combine{}, kIdentity, ctx_.cuda_stream));
}

stats_result stats_ctx::run(core::span2d<float> data, core::span2d<bool> mask)
{
  run_async(data, mask);

  CHECK_CUDA(cudaMemcpyAsync(&h_state_, d_state_.get(), sizeof(stats_acc), cudaMemcpyDeviceToHost, ctx_.cuda_stream));
  ctx_.wait();

  if (h_state_.count == 0) return {-FLT_MAX, 0.f};

  const float inv_n = 1.f / static_cast<float>(h_state_.count);
  const float mean = h_state_.sum * inv_n;
  const float msq = h_state_.sum_sq * inv_n;
  return {h_state_.max_v, std::sqrt(std::max(msq - mean * mean, 0.f))};
}

}  // namespace fast_deconv::matrix
