// #pragma once
//
// #include <fmt/base.h>
// #include <thrust/iterator/counting_iterator.h>
// #include <thrust/iterator/transform_iterator.h>
//
// #include <cmath>
// #include <cub/cub.cuh>
// #include <cuda/std/cstdint>
// #include <fast_deconv/core/span_types.hpp>
// #include <fast_deconv/core/resources.hpp>
// #include <fast_deconv/util/cuda_macros.hpp>
//
// namespace fast_deconv::matrix::detail {
//
// struct masking_op {
//   const float* data;
//   const bool* mask;
//   size_t mask_size;
//
//   __device__ __forceinline__ float operator()(const int& i) const
//   {
//     return mask[i % mask_size] ? data[i] : -INFINITY;
//   }
// };
//
// struct masking_op_abs {
//   const float* data;
//   const bool* mask;
//   size_t mask_size;
//
//   __device__ __forceinline__ float operator()(const int& i) const
//   {
//     return mask[i % mask_size] ? fabs(data[i]) : -INFINITY;
//   }
// };
//
// template <typename MaskingOpT>
// void argmax_async(core::stream_resources& resources, const float* data, const bool* mask,
//                   size_t size, size_t mask_size, float* d_max_out, uint* d_index_out)
// {
//   MaskingOpT op{data, mask, mask_size};
//   thrust::counting_iterator<int> counting_iter{0};
//   auto masked_iter = thrust::make_transform_iterator(counting_iter, op);
//
//   auto stream = resources.cuda_stream;
//
//   size_t temp_storage_bytes = 0;
//   CHECK_CUDA(cub::DeviceReduce::ArgMax(nullptr, temp_storage_bytes, masked_iter, d_max_out,
//                                        d_index_out, size, stream));
//
//   resources.alloc_device(temp_storage_bytes);
//
//   CHECK_CUDA(cub::DeviceReduce::ArgMax(resources.device_workspace, temp_storage_bytes, masked_iter,
//                                        d_max_out, d_index_out, size, stream));
// }
//
// }  // namespace fast_deconv::matrix::detail
