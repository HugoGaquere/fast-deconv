#pragma once

#include "fast_deconv/core/span_types.hpp"
#include <cub/cub.cuh>
#include <cuda/std/cstdint>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>

#include <emu/cuda/device/mdspan.hpp>
#include <fast_deconv/core/stream_resources.hpp>
#include <fast_deconv/util/cuda_macros.hpp>
#include <fmt/base.h>

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

struct span_masking_op {
  const core::device_span2d_f data;
  const core::device_span2d_b mask;
  const int cols;

  __device__ __forceinline__ float operator()(const int& i) const
  {
    const int row = i / cols;
    const int col = i % cols;

    return mask(row, col) ? data(row, col) : -INFINITY;
  }
};

struct span_masking_op_abs {
  const core::device_span2d_f data;
  const core::device_span2d_b mask;
  const int cols;

  __device__ __forceinline__ float operator()(const int& i) const
  {
    const int row = i / cols;
    const int col = i % cols;

    return mask(row, col) ? abs(data(row, col)) : -INFINITY;
  }
};

template <typename MaskingOpT>
void argmax_async(core::stream_resources& resources,
                  const float* data,
                  const bool* mask,
                  size_t size,
                  float* d_max_out,
                  uint* d_index_out)
{
  MaskingOpT op{data, mask};
  thrust::counting_iterator<int> counting_iter{0};
  auto masked_iter = thrust::make_transform_iterator(counting_iter, op);

  auto stream = resources.stream;

  size_t temp_storage_bytes = 0;
  CHECK_CUDA(cub::DeviceReduce::ArgMax(
    nullptr, temp_storage_bytes, masked_iter, d_max_out, d_index_out, size, stream));

  resources.alloc_device(temp_storage_bytes);

  CHECK_CUDA(cub::DeviceReduce::ArgMax(resources.device_workspace,
                                       temp_storage_bytes,
                                       masked_iter,
                                       d_max_out,
                                       d_index_out,
                                       size,
                                       stream));
}

template <typename MaskingOpT>
void argmax_async(core::stream_resources& resources,
                  float* d_max_out,
                  uint* d_index_out,
                  const core::device_span2d_f& data,
                  const core::device_span2d_b& mask)
{
  int rows = static_cast<int>(data.extent(0));
  int cols = static_cast<int>(data.extent(1));
  int size = rows * cols;
  MaskingOpT op{data, mask, cols};
  thrust::counting_iterator<int> counting_iter{0};
  auto masked_iter = thrust::make_transform_iterator(counting_iter, op);

  auto stream = resources.stream;

  size_t temp_storage_bytes = 0;

  CHECK_CUDA(cub::DeviceReduce::ArgMax(
    nullptr, temp_storage_bytes, masked_iter, d_max_out, d_index_out, size, stream));

  resources.alloc_device(temp_storage_bytes);

  CHECK_CUDA(cub::DeviceReduce::ArgMax(resources.device_workspace,
                                       temp_storage_bytes,
                                       masked_iter,
                                       d_max_out,
                                       d_index_out,
                                       size,
                                       stream));
}

}  // namespace fast_deconv::matrix::detail
