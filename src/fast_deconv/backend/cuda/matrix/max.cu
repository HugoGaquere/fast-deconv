#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>

#include <cassert>
#include <cfloat>
#include <cub/device/device_reduce.cuh>
#include <fast_deconv/matrix/max.hpp>
#include <fast_deconv/util/cuda_macros.hpp>

namespace {

/**
 * @brief   Functor for thrust transform iterator: applies mask and optional abs.
 * @details Returns -FLT_MAX for masked pixels so they are excluded from the max
 *          reduction. When @p use_abs is true, returns fabsf of the value.
 */
struct masked_max_op {
  const float* data;
  const bool* mask;
  bool use_abs;

  __host__ __device__ __forceinline__ float operator()(int idx) const
  {
    if (mask[idx]) return -FLT_MAX;
    return use_abs ? fabsf(data[idx]) : data[idx];
  }
};

}  // namespace

namespace fast_deconv::matrix {

float max(const core::exec_ctx& ctx, core::span2d<float> data, core::span2d<bool> mask, bool use_abs)
{
  assert(data.is_exhaustive() && mask.is_exhaustive());
  assert(data.extent(0) == mask.extent(0) && data.extent(1) == mask.extent(1));

  auto cuda_stream = ctx.cuda_stream;
  const int n = static_cast<int>(data.size());

  thrust::counting_iterator<int> counting(0);
  auto iter = thrust::make_transform_iterator(counting, masked_max_op{data.data_handle(), mask.data_handle(), use_abs});

  auto d_out = ctx.alloc_ptr_async<float>(1);

  size_t temp_bytes = 0;
  cub::DeviceReduce::Max(nullptr, temp_bytes, iter, d_out.get(), n, cuda_stream);
  auto d_temp = ctx.alloc_ptr_async<std::byte>(temp_bytes);
  cub::DeviceReduce::Max(d_temp.get(), temp_bytes, iter, d_out.get(), n, cuda_stream);

  float h_result;
  CHECK_CUDA(cudaMemcpyAsync(&h_result, d_out.get(), sizeof(float), cudaMemcpyDeviceToHost, cuda_stream));
  ctx.sync();

  return h_result;
}

}  // namespace fast_deconv::matrix
