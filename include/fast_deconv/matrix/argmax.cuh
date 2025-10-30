#pragma once

#include "fast_deconv/util/cuda_macros.hpp"

#include <cuda/std/mdspan>

#include <fast_deconv/core/span_types.hpp>
#include <fast_deconv/core/stream_resources.hpp>
#include <fast_deconv/matrix/detail/argmax.cuh>

#include <utility>

namespace fast_deconv::matrix {

inline void argmax_async(std::pair<int, float>* out,
                         const float* data,
                         const bool* mask,
                         size_t size,
                         bool use_abs,
                         core::stream_resources& resources,
                         cudaStream_t stream_override = nullptr)
{
  auto stream = stream_override != nullptr ? stream_override : resources.stream;

  if (use_abs)
    detail::argmax_async<detail::masking_op_abs>(resources, data, mask, size, out, stream);
  else
    detail::argmax_async<detail::masking_op>(resources, data, mask, size, out, stream);
}

template <typename T1, typename T2, typename ExtentsA, typename ExtentsB>
void argmax_async(std::pair<int, float>* out,
                  cuda::std::mdspan<const T1, ExtentsA> data,
                  cuda::std::mdspan<const T2, ExtentsB> mask,
                  bool use_abs,
                  core::stream_resources& resources,
                  cudaStream_t stream_override = nullptr)
{
  fast_deconv::matrix::argmax_async(
    out, data.data_handle(), mask.data_handle(), data.size(), use_abs, resources, stream_override);
}

std::pair<int, float> argmax(
  const float* data, const bool* mask, size_t size, bool use_abs, core::stream_resources& resources)
{
  auto stream = resources.stream;

  resources.alloc_device_output(sizeof(std::pair<int, float>), stream);
  auto* device_out = static_cast<std::pair<int, float>*>(resources.device_output_workspace);

  resources.alloc_host(sizeof(std::pair<int, float>));
  auto* host_out = static_cast<std::pair<int, float>*>(resources.host_workspace);

  argmax_async(device_out, data, mask, size, use_abs, resources, stream);
  CHECK_CUDA(cudaMemcpyAsync(
    host_out, device_out, sizeof(std::pair<int, float>), cudaMemcpyDeviceToHost, stream));
  CHECK_CUDA(cudaStreamSynchronize(stream));

  return *host_out;
}

template <typename T1, typename T2, typename ExtentsA, typename ExtentsB>
std::pair<int, float> argmax(cuda::std::mdspan<T1, ExtentsA> data,
                             cuda::std::mdspan<T2, ExtentsB> mask,
                             bool use_abs,
                             core::stream_resources& resources)
{
  return fast_deconv::matrix::argmax(
    data.data_handle(), mask.data_handle(), data.size(), use_abs, resources);
}

template <typename T>
std::pair<int, float> test_argmax(core::span_2d<T> data,
                                  core::span_2d<bool> mask,
                                  bool use_abs,
                                  core::stream_resources& resources)
{
  return fast_deconv::matrix::argmax(
    data.data_handle(), mask.data_handle(), data.size(), use_abs, resources);
}

}  // namespace fast_deconv::matrix
