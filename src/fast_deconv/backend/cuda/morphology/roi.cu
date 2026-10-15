#include <thrust/iterator/counting_iterator.h>
#include <thrust/transform_reduce.h>

#include <cstdint>
#include <fast_deconv/morphology/roi.hpp>

namespace fast_deconv::morphology {

common::roi compute_mask_roi(const core::exec_ctx& ctx, core::span2d<bool> data)
{
  bool* data_ptr = data.data_handle();
  const std::int64_t W = data.extent(1);
  const std::int64_t n = data.mapping().required_span_size();

  auto transform = [=] __device__(std::int64_t i) -> common::roi {
    if (!data_ptr[i]) return {INT64_MAX, 0, INT64_MAX, 0};
    const std::int64_t r = i / W, c = i % W;
    return {r, r + 1, c, c + 1};
  };

  auto combine = [] __device__(common::roi a, common::roi b) -> common::roi {
    return {min(a.rmin, b.rmin), max(a.rmax, b.rmax), min(a.cmin, b.cmin), max(a.cmax, b.cmax)};
  };

  common::roi result = thrust::transform_reduce(
      thrust::cuda::par.on(ctx.cuda_stream), thrust::counting_iterator<std::int64_t>(0),
      thrust::counting_iterator<std::int64_t>(n), transform, common::roi{INT64_MAX, 0, INT64_MAX, 0}, combine);

  // Nothing raised rmax, so the mask was empty: report an empty box, not the identity.
  if (result.rmax == 0) return {};
  return result;
}

}  // namespace fast_deconv::morphology
