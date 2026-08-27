#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cub/device/device_reduce.cuh>
#include <fast_deconv/matrix/rms.hpp>
#include <fast_deconv/util/cuda_macros.hpp>

namespace {

/**
 * @brief   Functor that returns the pixel value for unmasked pixels, 0 otherwise.
 */
struct masked_value_op {
  const float* data;
  const bool* mask;

  __host__ __device__ __forceinline__ float operator()(int idx) const { return mask[idx] ? 0.0f : data[idx]; }
};

/**
 * @brief   Functor that returns the squared pixel value for unmasked pixels, 0 otherwise.
 */
struct masked_value_sq_op {
  const float* data;
  const bool* mask;

  __host__ __device__ __forceinline__ float operator()(int idx) const
  {
    if (mask[idx]) return 0.0f;
    float v = data[idx];
    return v * v;
  }
};

/**
 * @brief   Functor that returns 1 for unmasked pixels, 0 otherwise.
 */
struct masked_count_op {
  const bool* mask;

  __host__ __device__ __forceinline__ int operator()(int idx) const { return mask[idx] ? 0 : 1; }
};

}  // namespace

namespace fast_deconv::matrix {

float rms(const core::exec_ctx& ctx, core::span2d<float> data, core::span2d<bool> mask)
{
  assert(data.is_exhaustive() && mask.is_exhaustive());
  assert(data.extent(0) == mask.extent(0) && data.extent(1) == mask.extent(1));

  auto cuda_stream = ctx.cuda_stream;
  const int n = static_cast<int>(data.size());
  thrust::counting_iterator<int> counting(0);

  auto d_sum = ctx.alloc_ptr_async<float>(1);
  auto d_sum_sq = ctx.alloc_ptr_async<float>(1);
  auto d_count = ctx.alloc_ptr_async<int>(1);

  // Sum of unmasked values
  auto sum_iter = thrust::make_transform_iterator(counting, masked_value_op{data.data_handle(), mask.data_handle()});
  size_t temp_bytes_sum = 0;
  cub::DeviceReduce::Sum(nullptr, temp_bytes_sum, sum_iter, d_sum.get(), n, cuda_stream);
  auto d_temp_sum = ctx.alloc_ptr_async<std::byte>(temp_bytes_sum);
  cub::DeviceReduce::Sum(d_temp_sum.get(), temp_bytes_sum, sum_iter, d_sum.get(), n, cuda_stream);

  // Sum of squared unmasked values
  auto sq_iter = thrust::make_transform_iterator(counting, masked_value_sq_op{data.data_handle(), mask.data_handle()});
  size_t temp_bytes_sq = 0;
  cub::DeviceReduce::Sum(nullptr, temp_bytes_sq, sq_iter, d_sum_sq.get(), n, cuda_stream);
  auto d_temp_sq = ctx.alloc_ptr_async<std::byte>(temp_bytes_sq);
  cub::DeviceReduce::Sum(d_temp_sq.get(), temp_bytes_sq, sq_iter, d_sum_sq.get(), n, cuda_stream);

  // Count of unmasked pixels
  auto count_iter = thrust::make_transform_iterator(counting, masked_count_op{mask.data_handle()});
  size_t temp_bytes_cnt = 0;
  cub::DeviceReduce::Sum(nullptr, temp_bytes_cnt, count_iter, d_count.get(), n, cuda_stream);
  auto d_temp_cnt = ctx.alloc_ptr_async<std::byte>(temp_bytes_cnt);
  cub::DeviceReduce::Sum(d_temp_cnt.get(), temp_bytes_cnt, count_iter, d_count.get(), n, cuda_stream);

  float h_sum, h_sum_sq;
  int h_count;
  CHECK_CUDA(cudaMemcpyAsync(&h_sum, d_sum.get(), sizeof(float), cudaMemcpyDeviceToHost, cuda_stream));
  CHECK_CUDA(cudaMemcpyAsync(&h_sum_sq, d_sum_sq.get(), sizeof(float), cudaMemcpyDeviceToHost, cuda_stream));
  CHECK_CUDA(cudaMemcpyAsync(&h_count, d_count.get(), sizeof(int), cudaMemcpyDeviceToHost, cuda_stream));
  ctx.wait();

  if (h_count == 0) return 0.0f;
  float mean = h_sum / static_cast<float>(h_count);
  float var = h_sum_sq / static_cast<float>(h_count) - mean * mean;
  return std::sqrt(std::max(var, 0.0f));
}

}  // namespace fast_deconv::matrix
