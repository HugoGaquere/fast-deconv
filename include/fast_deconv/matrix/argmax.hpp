#pragma once

#include "fast_deconv/util/cuda_macros.hpp"

#include <cuda/std/mdspan>

#include <fast_deconv/core/span_types.hpp>
#include <fast_deconv/core/stream_resources.hpp>
#include <fast_deconv/matrix/detail/argmax.hpp>

#include <utility>

namespace fast_deconv::matrix {

inline void argmax_async(std::pair<int, float>* out,
                         const float* data,
                         const bool* mask,
                         size_t size,
                         bool use_abs,
                         core::stream_resources& resources)
{
  if (use_abs)
    detail::argmax_async<detail::masking_op_abs>(resources, data, mask, size, out);
  else
    detail::argmax_async<detail::masking_op>(resources, data, mask, size, out);
}

std::pair<int, float> argmax(
  const float* data, const bool* mask, size_t size, bool use_abs, core::stream_resources& resources)
{
  std::pair<int, float> out{-1, 0};
  auto stream = resources.stream;
  resources.alloc_device_output(sizeof(std::pair<int, float>));
  auto* device_out = static_cast<std::pair<int, float>*>(resources.device_output_workspace);

  argmax_async(device_out, data, mask, size, use_abs, resources);

  CHECK_CUDA(cudaMemcpyAsync(
    &out, device_out, sizeof(std::pair<int, float>), cudaMemcpyDeviceToHost, stream));
  CHECK_CUDA(cudaStreamSynchronize(stream));

  return out;
}

}  // namespace fast_deconv::matrix
