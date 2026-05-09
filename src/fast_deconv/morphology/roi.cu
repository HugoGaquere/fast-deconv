#include <thrust/iterator/counting_iterator.h>
#include <thrust/transform_reduce.h>

#include <fast_deconv/morphology/roi.hpp>

namespace fast_deconv::morphology {

roi compute_mask_roi(const core::stream_resources& stream, core::device_span2d<bool> data)
{
  bool* data_ptr = data.data_handle();
  int H = data.extent(0);
  int W = data.extent(1);

  auto transform = [=] __device__(int i) -> roi {
    if (!data_ptr[i]) return {INT_MAX, -1, INT_MAX, -1};
    int r = i / W, c = i % W;
    return {r, r, c, c};
  };

  auto combine = [] __device__(roi a, roi b) -> roi {
    return {min(a.xmin, b.xmin), max(a.xmax, b.xmax), min(a.ymin, b.ymin), max(a.ymax, b.ymax)};
  };

  roi result = thrust::transform_reduce(thrust::cuda::par.on(stream.cuda_stream), thrust::counting_iterator<int>(0),
                                        thrust::counting_iterator<int>(H * W), transform, roi{INT_MAX, -1, INT_MAX, -1},
                                        combine);

  return result;
}

}  // namespace fast_deconv::morphology
