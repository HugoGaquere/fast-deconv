#pragma once

#include <cassert>
#include <emu/cuda/device/mdspan.hpp>
#include <fast_deconv/core/concepts.hpp>
#include <fast_deconv/core/stream_resources.hpp>
#include <fast_deconv/matrix/detail/argmax.cuh>
#include <fast_deconv/util/cuda_macros.hpp>

namespace fast_deconv::matrix {

namespace cpts = fast_deconv::core::cpts;

inline void argmax_async(float* d_max_out, uint* d_index_out, const float* data, const bool* mask,
                         size_t size, size_t mask_size, bool use_abs,
                         core::stream_resources& resources)
{
  if (use_abs)
    detail::argmax_async<detail::masking_op_abs>(resources, data, mask, size, mask_size, d_max_out,
                                                 d_index_out);
  else
    detail::argmax_async<detail::masking_op>(resources, data, mask, size, mask_size, d_max_out,
                                             d_index_out);
}

inline std::pair<int, float> argmax(const float* data, const bool* mask, size_t size,
                                    size_t mask_size, bool use_abs,
                                    core::stream_resources& resources)
{
  auto stream = resources.stream;

  float* d_max_out;   // memory for the maximum value
  uint* d_index_out;  // memory for the index of the returned value
  CHECK_CUDA(cudaMallocAsync(reinterpret_cast<void**>(&d_max_out), sizeof(float), stream));
  CHECK_CUDA(cudaMallocAsync(reinterpret_cast<void**>(&d_index_out), sizeof(uint), stream));

  argmax_async(d_max_out, d_index_out, data, mask, size, mask_size, use_abs, resources);

  float h_max_out;
  uint h_index_out;
  CHECK_CUDA(cudaMemcpyAsync(&h_max_out, d_max_out, sizeof(float), cudaMemcpyDeviceToHost, stream));
  CHECK_CUDA(
      cudaMemcpyAsync(&h_index_out, d_index_out, sizeof(uint), cudaMemcpyDeviceToHost, stream));

  resources.sync();

  return {h_index_out, h_max_out};
}

template <cpts::mdspan Data, cpts::mdspan Mask>
inline std::pair<int, float> argmax(const Data& data, const Mask& mask, bool use_abs,
                                    core::stream_resources& resources)
{
  static_assert(!cpts::is_layout_stride<Data>, "argmax: layout_stride is not implemented yet.");
  static_assert(!cpts::is_layout_stride<Mask>, "argmax: layout_stride is not implemented yet.");
  assert(data.size() % mask.size() == 0 && "argmax: data size must be a multiple of mask size");
  return argmax(data.data_handle(), mask.data_handle(), data.size(), mask.size(), use_abs,
                resources);
}

}  // namespace fast_deconv::matrix
