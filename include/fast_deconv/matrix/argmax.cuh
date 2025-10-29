#pragma once

#include "fast_deconv/util/cuda_macros.hpp"

#include <cuda/std/mdspan>

#include <fast_deconv/core/stream_resources.hpp>
#include <fast_deconv/matrix/detail/argmax.cuh>

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

template <typename T1, typename T2, typename ExtentsA, typename ExtentsB>
void argmax_async(std::pair<int, float>* out,
                  cuda::std::mdspan<const T1, ExtentsA> data,
                  cuda::std::mdspan<const T2, ExtentsB> mask,
                  bool use_abs,
                  core::stream_resources& resources)
{
  fast_deconv::matrix::argmax_async(
    out, data.data_handle(), mask.data_handle(), data.size(), use_abs, resources);
}

std::pair<int, float> argmax(
  const float* data, const bool* mask, size_t size, bool use_abs, core::stream_resources& resources)
{
  std::pair<int, float>* res;
  CHECK_CUDA(cudaMallocManaged(reinterpret_cast<void**>(&res), sizeof(std::pair<int, float>)));

  argmax_async(res, data, mask, size, use_abs, resources);
  CHECK_CUDA(cudaStreamSynchronize(resources.stream));

  auto out = *res;
  CHECK_CUDA(cudaFree(res));

  return out;
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

}  // namespace fast_deconv::matrix
