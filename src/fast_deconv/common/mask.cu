#include <cassert>
#include <fast_deconv/common/mask.hpp>
#include <fast_deconv/util/cuda_macros.hpp>

namespace fast_deconv::kernel {

__global__ void mask_and_abs_kernel(float* data, const bool* mask, float fill_value, bool abs, int n_per_batch,
                                    int batch_stride)
{
  float* d = data + blockIdx.y * batch_stride;
  const int stride = blockDim.x * gridDim.x;
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n_per_batch; i += stride) {
    if (mask[i])
      d[i] = fill_value;
    else if (abs)
      d[i] = fabsf(d[i]);
  }
}

__global__ void mask_less_than_threshold_kernel(float* data, float threshold, float fill_value, int n_per_batch,
                                                int batch_stride)
{
  float* d = data + blockIdx.y * batch_stride;
  const int stride = blockDim.x * gridDim.x;
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n_per_batch; i += stride) {
    if (data[i] < threshold) d[i] = fill_value;
  }
}
}  // namespace fast_deconv::kernel

namespace fast_deconv::common {

void mask_and_abs_async(const core::stream_resources& stream_res, core::device_span2d<float> data,
                        core::device_span2d<bool> mask, float fill_value, bool abs)
{
  assert(data.is_exhaustive() && mask.is_exhaustive());
  assert(data.extent(0) == mask.extent(0) && data.extent(1) == mask.extent(1));
  const int n = static_cast<int>(data.size());
  dim3 grid(CEIL_DIV(n, 256), 1);
  kernel::mask_and_abs_kernel<<<grid, 256, 0, stream_res.cuda_stream>>>(data.data_handle(), mask.data_handle(),
                                                                        fill_value, abs, n, 0);
}

void mask_and_abs_async(const core::stream_resources& stream_res, core::device_span3d<float> data,
                        core::device_span2d<bool> mask, float fill_value, bool abs)
{
  assert(data.is_exhaustive() && mask.is_exhaustive());
  assert(data.extent(1) == mask.extent(0) && data.extent(2) == mask.extent(1));
  const int n_per_batch = static_cast<int>(mask.size());
  const int nbatch = static_cast<int>(data.extent(0));
  const int batch_stride = static_cast<int>(data.stride(0));
  dim3 grid(CEIL_DIV(n_per_batch, 256), nbatch);
  kernel::mask_and_abs_kernel<<<grid, 256, 0, stream_res.cuda_stream>>>(data.data_handle(), mask.data_handle(),
                                                                        fill_value, abs, n_per_batch, batch_stride);
}

void mask_less_than_threshold(const core::stream_resources& stream_res, core::device_span2d<float> data,
                              float threshold, float fill_value)
{
  assert(data.is_exhaustive());
  const int n = static_cast<int>(data.size());
  dim3 grid(CEIL_DIV(n, 256), 1);
  kernel::mask_less_than_threshold_kernel<<<grid, 256, 0, stream_res.cuda_stream>>>(data.data_handle(), threshold,
                                                                                    fill_value, n, 0);
}

}  // namespace fast_deconv::common
