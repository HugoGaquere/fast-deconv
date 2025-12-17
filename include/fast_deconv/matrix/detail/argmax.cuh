#pragma once

#include <cub/cub.cuh>
#include <cuda/std/cstdint>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>

#include <fast_deconv/core/span_types.hpp>
#include <fast_deconv/core/stream_resources.hpp>
#include <fast_deconv/util/cuda_macros.hpp>
#include <fmt/base.h>
#include <cmath>

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
    return mask[i] ? fabs(data[i]) : -INFINITY;
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

// =========== MDSPAN: WIP ===========

// template <typename DataMdspan, typename MaskMdspan>
// struct span_masking_op {
//   DataMdspan data;
//   MaskMdspan mask;
//
//   __device__ __forceinline__ float operator()(int i) const
//   {
//     int cols = (int)data.extent(1);
//     const int row = i / cols;
//     const int col = i % cols;
//
//     printf("== data.size=%d i=%d\n", (int)data.size(), i);
//     printf("== i=%d, row=%d, col=%d cols=%d\n", i, row, col, cols);
//     // return 0.f;
//     return mask(row, col) ? data(row, col) : -INFINITY;
//   }
// };
//
//
// template <typename DataMdspan, typename MaskMdspan>
// struct span_masking_op_abs {
//   DataMdspan data;
//   MaskMdspan mask;
//
//   __device__ __forceinline__ float operator()(int i) const
//   {
//     int cols = (int)data.extent(1);
//     const int row = i / cols;
//     const int col = i % cols;
//
//     // if (i >= data.size()) return -INFINITY;
//
//     // return 0.f;
//     return mask(row, col) ? fabs(data(row, col)) : -INFINITY;
//   }
// };
//

// template <typename DataMdspan, typename MaskMdspan>
// __global__ void argmax_kernel(DataMdspan data, MaskMdspan mask) {
//   const uint tid = blockIdx.x * blockDim.x + threadIdx.x;
//   const uint size = static_cast<uint>(data.size());
//   if (tid >= size) return;
//
//   const uint cols = static_cast<uint>(data.extent(1));
//   const uint row = tid / cols;
//   const uint col = tid % cols;
//   float val = data(row, col);
//
//   printf("tid=%d, data[tid]=%f \n", tid, val);
// }

// template <typename MaskingOpT, typename DataMdspan, typename MaskMdspan>
// void argmax_mdspan_async(core::stream_resources& resources,
//                          float* d_max_out,
//                          uint* d_index_out,
//                          const DataMdspan& data,
//                          const MaskMdspan& mask)
// {
//   int rows = static_cast<int>(data.extent(0));
//   int cols = static_cast<int>(data.extent(1));
//   fmt::println("{} {}", rows, cols);
//   int size = rows * cols;
//
//   argmax_kernel<<<1, 100, 0, resources.stream>>>(data, mask);
//   resources.sync();
//
//   // auto op = MaskingOpT{data, mask};
//   empty_op op{data};
//
//   thrust::counting_iterator<int> counting_iter{0};
//   auto masked_iter = thrust::make_transform_iterator(counting_iter, op);
//
//   auto stream = resources.stream;
//
//   size_t temp_storage_bytes = 0;
//
//   CHECK_CUDA(cub::DeviceReduce::ArgMax(
//     nullptr, temp_storage_bytes, masked_iter, d_max_out, d_index_out, size, stream));
//
//   resources.alloc_device(temp_storage_bytes);
//
//   CHECK_CUDA(cub::DeviceReduce::ArgMax(resources.device_workspace,
//                                        temp_storage_bytes,
//                                        masked_iter,
//                                        d_max_out,
//                                        d_index_out,
//                                        size,
//                                        stream));
// }

}  // namespace fast_deconv::matrix::detail
