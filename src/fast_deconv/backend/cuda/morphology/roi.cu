#include <thrust/iterator/counting_iterator.h>
#include <thrust/transform_reduce.h>

#include <fast_deconv/morphology/roi.hpp>

namespace fast_deconv::morphology {

common::roi compute_mask_roi(const core::exec_ctx& ctx, core::span2d<bool> data)
{
  bool* data_ptr = data.data_handle();
  int H = data.extent(0);
  int W = data.extent(1);

  auto transform = [=] __device__(int i) -> common::roi {
    if (!data_ptr[i]) return {INT_MAX, 0, INT_MAX, 0};
    int r = i / W, c = i % W;
    return {r, r + 1, c, c + 1};
  };

  auto combine = [] __device__(common::roi a, common::roi b) -> common::roi {
    return {min(a.rmin, b.rmin), max(a.rmax, b.rmax), min(a.cmin, b.cmin), max(a.cmax, b.cmax)};
  };

  common::roi result = thrust::transform_reduce(
      thrust::cuda::par.on(ctx.cuda_stream), thrust::counting_iterator<int>(0), thrust::counting_iterator<int>(H * W),
      transform, common::roi{INT_MAX, 0, INT_MAX, 0}, combine);

  // Nothing raised rmax, so the mask was empty: report an empty box, not the identity.
  if (result.rmax == 0) return {};
  return result;
}

}  // namespace fast_deconv::morphology
