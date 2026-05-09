#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>

#include <algorithm>
#include <cassert>
#include <cfloat>
#include <cmath>
#include <cub/device/device_reduce.cuh>
#include <fast_deconv/matrix/stats.hpp>

namespace fast_deconv::matrix {

namespace {

/// Per-pixel transform: emits the identity contribution for masked pixels so
/// they can be combined into the reduction without affecting the result.
struct masked_stats_op {
  const float* data;
  const bool* mask;
  bool use_abs;

  __host__ __device__ __forceinline__ stats_acc operator()(int idx) const
  {
    if (mask[idx]) return {-FLT_MAX, 0.f, 0.f, 0};
    const float v = data[idx];
    return {use_abs ? fabsf(v) : v, v, v * v, 1};
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

stats_workspace::stats_workspace(const core::stream_resources& sr, size_t n, bool abs_)
    : stream_res(sr), n_elements(n), use_abs(abs_)
{
  d_state = stream_res.alloc_async<stats_acc>(1);

  // Query temp-storage bytes against the same iterator/op/state types we will
  // use at call time. The pointers held by the iterator are unused during the
  // sizing query.
  auto it_query = make_iter(nullptr, nullptr, use_abs);
  CHECK_CUDA(cub::DeviceReduce::Reduce(nullptr, temp_storage_bytes, it_query, d_state, static_cast<int>(n_elements),
                                       stats_combine{}, kIdentity, stream_res.cuda_stream));

  d_temp = stream_res.alloc_async<char>(temp_storage_bytes);
}

stats_workspace::~stats_workspace()
{
  if (d_temp) stream_res.free_async(d_temp);
  if (d_state) stream_res.free_async(d_state);
}

void compute_stats_async(stats_workspace& ws, core::device_span2d<float> data, core::device_span2d<bool> mask)
{
  assert(data.is_exhaustive() && mask.is_exhaustive());
  assert(data.extent(0) == mask.extent(0) && data.extent(1) == mask.extent(1));
  assert(data.size() == ws.n_elements);

  auto it = make_iter(data.data_handle(), mask.data_handle(), ws.use_abs);
  CHECK_CUDA(cub::DeviceReduce::Reduce(ws.d_temp, ws.temp_storage_bytes, it, ws.d_state,
                                       static_cast<int>(ws.n_elements), stats_combine{}, kIdentity,
                                       ws.stream_res.cuda_stream));
}

stats_result compute_stats(stats_workspace& ws, core::device_span2d<float> data, core::device_span2d<bool> mask)
{
  compute_stats_async(ws, data, mask);

  CHECK_CUDA(cudaMemcpyAsync(&ws.h_state, ws.d_state, sizeof(stats_acc), cudaMemcpyDeviceToHost,
                             ws.stream_res.cuda_stream));
  ws.stream_res.sync();

  if (ws.h_state.count == 0) return {-FLT_MAX, 0.f};

  const float inv_n = 1.f / static_cast<float>(ws.h_state.count);
  const float mean = ws.h_state.sum * inv_n;
  const float msq = ws.h_state.sum_sq * inv_n;
  return {ws.h_state.max_v, std::sqrt(std::max(msq - mean * mean, 0.f))};
}

}  // namespace fast_deconv::matrix
