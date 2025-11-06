#pragma once

#include <cub/cub.cuh>
#include <cub/util_allocator.cuh>
#include <cuda/std/cstdint>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>

#include <fast_deconv/core/stream_resources.hpp>
#include <fast_deconv/util/cuda_macros.hpp>

namespace fast_deconv::matrix::detail {

struct masking_op {
  const float* data;
  const bool* mask;

  __device__ __forceinline__ float operator()(const int& i) const
  {
    return mask[i] ? data[i] : -INFINITY;
  }
};

struct masking_op_abs {
  const float* data;
  const bool* mask;

  __device__ __forceinline__ float operator()(const int& i) const
  {
    return mask[i] ? abs(data[i]) : -INFINITY;
  }
};

template <typename MaskingOpT>
void argmax_async(core::stream_resources& resources,
                  const float* data,
                  const bool* mask,
                  size_t size,
                  std::pair<int, float>* out)
{
  MaskingOpT op{data, mask};
  thrust::counting_iterator<int> counting_iter{0};
  auto masked_iter = thrust::make_transform_iterator(counting_iter, op);

  auto stream      = resources.stream;
  auto* casted_out = reinterpret_cast<cub::KeyValuePair<int, float>*>(out);

  size_t temp_storage_bytes = 0;
  CHECK_CUDA(
    cub::DeviceReduce::ArgMax(nullptr, temp_storage_bytes, masked_iter, casted_out, size, stream));

  resources.alloc_device(temp_storage_bytes);

  CHECK_CUDA(cub::DeviceReduce::ArgMax(
    resources.device_workspace, temp_storage_bytes, masked_iter, casted_out, size, stream));
}

}  // namespace fast_deconv::matrix::detail
