#pragma once

#include "fast_deconv/util/cuda_macros.hpp"

#include <emu/cuda/device/mdspan.hpp>
#include <fast_deconv/core/stream_resources.hpp>
#include <fast_deconv/matrix/detail/argmax.cuh>

namespace fast_deconv::matrix {

inline void argmax_async(float* d_max_out,
                         uint* d_index_out,
                         const float* data,
                         const bool* mask,
                         size_t size,
                         bool use_abs,
                         core::stream_resources& resources)
{
  if (use_abs)
    detail::argmax_async<detail::masking_op_abs>(
      resources, data, mask, size, d_max_out, d_index_out);
  else
    detail::argmax_async<detail::masking_op>(resources, data, mask, size, d_max_out, d_index_out);
}

inline std::pair<int, float> argmax(
  const float* data, const bool* mask, size_t size, bool use_abs, core::stream_resources& resources)
{
  auto stream = resources.stream;

  float* d_max_out;   // memory for the maximum value
  uint* d_index_out;  // memory for the index of the returned value
  CHECK_CUDA(cudaMallocAsync(reinterpret_cast<void**>(&d_max_out), sizeof(float), stream));
  CHECK_CUDA(cudaMallocAsync(reinterpret_cast<void**>(&d_index_out), sizeof(uint), stream));

  argmax_async(d_max_out, d_index_out, data, mask, size, use_abs, resources);

  float h_max_out;
  uint h_index_out;
  CHECK_CUDA(cudaMemcpyAsync(&h_max_out, d_max_out, sizeof(float), cudaMemcpyDeviceToHost, stream));
  CHECK_CUDA(
    cudaMemcpyAsync(&h_index_out, d_index_out, sizeof(uint), cudaMemcpyDeviceToHost, stream));

  resources.sync();

  return {h_index_out, h_max_out};
}

template <typename DataMdspan, typename MaskMdspan>
inline std::pair<int, float> argmax(const DataMdspan& data,
                                    const MaskMdspan& mask,
                                    bool use_abs,
                                    core::stream_resources& resources)
{
  return argmax(data.data_handle(), mask.data_handle(), data.size(), use_abs, resources);
}

// template <typename DataMdspan, typename MaskMdspan>
// inline void argmax_async(float* d_max_out,
//                          uint* d_index_out,
//                          bool use_abs,
//                          core::stream_resources& resources,
//                          const DataMdspan& data,
//                          const MaskMdspan& mask)
// {
//   if (use_abs)
//     detail::argmax_mdspan_async<detail::span_masking_op_abs<DataMdspan, MaskMdspan>>(resources,
//     d_max_out, d_index_out, data, mask);
//   else
//     detail::argmax_mdspan_async<detail::span_masking_op<DataMdspan, MaskMdspan>>(resources,
//     d_max_out, d_index_out, data, mask);
// }

// template <typename DataMdspan, typename MaskMdspan>
// inline std::pair<int, float> argmax(bool use_abs,
//                                     core::stream_resources& resources,
//                                     DataMdspan& data,
//                                     MaskMdspan& mask)
// {
//   auto stream = resources.stream;
//
//   float* d_max_out;   // memory for the maximum value
//   uint* d_index_out;  // memory for the index of the returned value
//   CHECK_CUDA(cudaMallocAsync(reinterpret_cast<void**>(&d_max_out), sizeof(float), stream));
//   CHECK_CUDA(cudaMallocAsync(reinterpret_cast<void**>(&d_index_out), sizeof(uint), stream));
//
//   argmax_async(d_max_out, d_index_out, use_abs, resources, data, mask);
//
//   float h_max_out;
//   uint h_index_out;
//   CHECK_CUDA(cudaMemcpyAsync(&h_max_out, d_max_out, sizeof(float), cudaMemcpyDeviceToHost,
//   stream)); CHECK_CUDA(
//     cudaMemcpyAsync(&h_index_out, d_index_out, sizeof(uint), cudaMemcpyDeviceToHost, stream));
//
//   resources.sync();
//
//   return {h_index_out, h_max_out};
// }

}  // namespace fast_deconv::matrix
