#pragma once

#include <cub/cub.cuh>

#include <fast_deconv/core/stream_resources.hpp>

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
  // Create the transform iterator that applies masking on-the-fly
  cub::CountingInputIterator<int> counting_iter{0};
  MaskingOpT op{data, mask};
  cub::TransformInputIterator<float, MaskingOpT, cub::CountingInputIterator<int>> masked_iter{counting_iter,
                                                                                    op};

  // Cast out to cub::KeyValuePair
  auto out_as_keyvalue = reinterpret_cast<cub::KeyValuePair<int, float>*>(out);

  // First call to get temp storage size
  size_t temp_storage_bytes = 0;
  cub::DeviceReduce::ArgMax(resources.device_workspace,
                       temp_storage_bytes,
                       masked_iter,
                       out_as_keyvalue,
                       size,
                       resources.stream);
  resources.alloc_device(temp_storage_bytes);

  // Actual ArgMax call
  cub::DeviceReduce::ArgMax(resources.device_workspace,
                       temp_storage_bytes,
                       masked_iter,
                       out_as_keyvalue,
                       size,
                       resources.stream);
}

}  // namespace fast_deconv::matrix::detail
